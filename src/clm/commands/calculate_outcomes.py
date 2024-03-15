import argparse
import os
import numpy as np
import pandas as pd
import scipy.stats
from fcd_torch import FCD
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, RDKFingerprint
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.MolSurf import TPSA
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from rdkit import rdBase
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
from collections import defaultdict
from clm.functions import (
    read_file,
    seed_type,
    set_seed,
    clean_mol,
    # Functions for calculating metrics
    continuous_JSD,
    discrete_JSD,
    internal_diversity,
    external_diversity,
    internal_nn,
    external_nn,
    pct_rotatable_bonds,
    pct_stereocenters,
)

rdBase.DisableLog("rdApp.error")
fscore = npscorer.readNPModel()


def add_args(parser):
    parser.add_argument(
        "--train_file", type=str, help="Training csv file with smiles as a column."
    )
    parser.add_argument(
        "--sampled_file",
        type=str,
        help="Sampled csv file with smiles as a column, or a text file with one SMILES per line.",
    )
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_orig_mols", type=int, default=10000)
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed"
    )
    return parser


def safe_qed(mol):
    try:
        return qed(mol)
    except OverflowError:
        return None


def safe_sascorer(mol):
    try:
        return sascorer.calculateScore(mol)
    except (OverflowError, ZeroDivisionError):
        return None


molecular_properties = {
    "elements": lambda mol: [atom.GetSymbol() for atom in mol.GetAtoms()],
    "mws": lambda mol: Descriptors.MolWt(mol),
    "logp": lambda mol: Descriptors.MolLogP(mol),
    "tcs": lambda mol: BertzCT(mol),
    "tpsa": lambda mol: TPSA(mol),
    "qed": lambda mol: safe_qed(mol),
    "rings1": lambda mol: Lipinski.RingCount(mol),
    "rings2": lambda mol: Lipinski.NumAliphaticRings(mol),
    "rings3": lambda mol: Lipinski.NumAromaticRings(mol),
    "SA": lambda mol: safe_sascorer(mol),
    "NP": lambda mol: npscorer.scoreMol(mol, fscore),
    "sp3": lambda mol: Lipinski.FractionCSP3(mol),
    "rot": lambda mol: pct_rotatable_bonds(mol),
    "stereo": lambda mol: pct_stereocenters(mol),
    "murcko": lambda mol: MurckoScaffoldSmiles(mol=mol),
    "donors": lambda mol: Lipinski.NumHDonors(mol),
    "acceptors": lambda mol: Lipinski.NumHAcceptors(mol),
    "fps": lambda mol: RDKFingerprint(mol),
}


def process_smiles(smiles, train_smiles=None, is_train=False):
    dict = defaultdict(list)
    dict["n_valid_mols"], dict["n_novel_mols"] = 0, 0

    for i, smile in enumerate(smiles, start=1):
        if (mol := clean_mol(smile)) is not None:
            dict["canonical"].append(Chem.MolToSmiles(mol))
            dict["n_valid_mols"] += 1

            # Only store novel smiles from the sampled file
            if is_train or (train_smiles is not None and smile not in train_smiles):
                for key, fun in molecular_properties.items():
                    dict[key].append(fun(mol))
                dict["n_novel_mols"] += 1

    dict["n_smiles"] = i
    dict["n_unique"] = len(set(dict["canonical"]))

    return dict


def calculate_probabilities(train_counts, gen_counts):
    keys = np.union1d(train_counts[0], gen_counts[0])
    n1, n2 = sum(train_counts[1]), sum(gen_counts[1])
    d1 = dict(zip(train_counts[0], train_counts[1]))
    d2 = dict(zip(gen_counts[0], gen_counts[1]))

    p1 = [(d1[key] / n1) if key in d1 else 0 for key in keys]
    p2 = [(d2[key] / n2) if key in d2 else 0 for key in keys]

    return p1, p2


def process_outcomes(train_dict, gen_dict, output_file, sampled_file):
    org_counts = np.unique(np.concatenate(train_dict["elements"]), return_counts=True)
    org_murcko_counts = np.unique(train_dict["murcko"], return_counts=True)
    gen_counts = np.unique(np.concatenate(gen_dict["elements"]), return_counts=True)
    gen_murcko_counts = np.unique(gen_dict["murcko"], return_counts=True)

    p1, p2 = calculate_probabilities(org_counts, gen_counts)
    p_m1, p_m2 = calculate_probabilities(org_murcko_counts, gen_murcko_counts)
    fcd = FCD(canonize=False)

    res = {
        "% valid": gen_dict["n_valid_mols"] / gen_dict["n_smiles"],
        "% novel": gen_dict["n_novel_mols"] / gen_dict["n_valid_mols"],
        "% unique": gen_dict["n_unique"] / gen_dict["n_valid_mols"],
        "KL divergence, atoms": scipy.stats.entropy(p2, p1),
        "Jensen-Shannon distance, atoms": jensenshannon(p2, p1),
        "Wasserstein distance, atoms": wasserstein_distance(p2, p1),
        "Jensen-Shannon distance, MWs": continuous_JSD(
            gen_dict["mws"], train_dict["mws"]
        ),
        "Jensen-Shannon distance, logP": continuous_JSD(
            gen_dict["logp"], train_dict["logp"]
        ),
        "Jensen-Shannon distance, Bertz TC": continuous_JSD(
            gen_dict["tcs"], train_dict["tcs"]
        ),
        "Jensen-Shannon distance, QED": continuous_JSD(
            gen_dict["qed"], train_dict["qed"]
        ),
        "Jensen-Shannon distance, TPSA": continuous_JSD(
            gen_dict["tpsa"], train_dict["tpsa"]
        ),
        "Internal diversity": internal_diversity(gen_dict["fps"]),
        "External diversity": external_diversity(gen_dict["fps"], train_dict["fps"]),
        "Internal nearest-neighbor Tc": internal_nn(gen_dict["fps"]),
        "External nearest-neighbor Tc": external_nn(gen_dict["fps"], train_dict["fps"]),
        "Jensen-Shannon distance, # of rings": discrete_JSD(
            gen_dict["rings1"], train_dict["rings1"]
        ),
        "Jensen-Shannon distance, # of aliphatic rings": discrete_JSD(
            gen_dict["rings2"], train_dict["rings2"]
        ),
        "Jensen-Shannon distance, # of aromatic rings": discrete_JSD(
            gen_dict["rings3"], train_dict["rings3"]
        ),
        "Jensen-Shannon distance, SA score": continuous_JSD(
            gen_dict["SA"], train_dict["SA"]
        ),
        "Jensen-Shannon distance, NP score": continuous_JSD(
            gen_dict["NP"], train_dict["NP"]
        ),
        "Jensen-Shannon distance, % sp3 carbons": continuous_JSD(
            gen_dict["sp3"], train_dict["sp3"]
        ),
        "Jensen-Shannon distance, % rotatable bonds": continuous_JSD(
            gen_dict["rot"], train_dict["rot"]
        ),
        "Jensen-Shannon distance, % stereocenters": continuous_JSD(
            gen_dict["stereo"], train_dict["stereo"]
        ),
        "Jensen-Shannon distance, Murcko scaffolds": jensenshannon(p_m2, p_m1),
        "Jensen-Shannon distance, hydrogen donors": discrete_JSD(
            gen_dict["donors"], train_dict["donors"]
        ),
        "Jensen-Shannon distance, hydrogen acceptors": discrete_JSD(
            gen_dict["acceptors"], train_dict["acceptors"]
        ),
        "Frechet ChemNet distance": fcd(gen_dict["canonical"], train_dict["canonical"]),
    }

    res = pd.DataFrame(list(res.items()), columns=["outcome", "value"])
    res["input_file"] = os.path.basename(sampled_file)
    res.to_csv(
        output_file,
        mode="a+",
        index=False,
        compression="gzip" if str(output_file).endswith(".gz") else None,
    )

    res.reset_index(inplace=True, drop=True)
    return res


def get_dicts(train_file, sampled_file, max_orig_mols, seed):
    set_seed(seed)

    gen_smiles = read_file(
        sampled_file, max_lines=max_orig_mols, stream=True, smile_only=True
    )
    train_smiles = read_file(train_file, smile_only=True)

    train_dict = process_smiles(train_smiles, is_train=True)
    gen_dict = process_smiles(gen_smiles, train_smiles=set(train_smiles))

    return train_dict, gen_dict


def calculate_outcomes(train_file, sampled_file, output_file, max_orig_mols, seed):
    train_dict, gen_dict = get_dicts(train_file, sampled_file, max_orig_mols, seed)
    return process_outcomes(train_dict, gen_dict, output_file, sampled_file)


def main(args):
    calculate_outcomes(
        train_file=args.train_file,
        sampled_file=args.sampled_file,
        output_file=args.output_file,
        max_orig_mols=args.max_orig_mols,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
