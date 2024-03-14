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
    continuous_JSD,
    discrete_JSD,
    internal_diversity,
    external_diversity,
    internal_nn,
    external_nn,
    pct_rotatable_bonds,
    pct_stereocenters,
    seed_type,
    set_seed,
    clean_mol,
)

rdBase.DisableLog("rdApp.error")


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


def calculate_molecular_properties(mol):
    fscore = npscorer.readNPModel()
    properties = {
        "mols": mol,
        "elements": [atom.GetSymbol() for atom in mol.GetAtoms()],
        "mws": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tcs": BertzCT(mol),
        "tpsa": TPSA(mol),
        "qed": safe_qed(mol),
        "rings1": Lipinski.RingCount(mol),
        "rings2": Lipinski.NumAliphaticRings(mol),
        "rings3": Lipinski.NumAromaticRings(mol),
        "SA": safe_sascorer(mol),
        "NP": npscorer.scoreMol(mol, fscore),
        "sp3": Lipinski.FractionCSP3(mol),
        "rot": pct_rotatable_bonds(mol),
        "stereo": pct_stereocenters(mol),
        "murcko": MurckoScaffoldSmiles(mol=mol),
        "donors": Lipinski.NumHDonors(mol),
        "acceptors": Lipinski.NumHAcceptors(mol),
        "fps": RDKFingerprint(mol),
    }
    return properties


def process_smiles(df, smiles, is_train=False):
    for smile in smiles:
        if (mol := clean_mol(smile)) is not None:
            properties = calculate_molecular_properties(mol)
            for key, value in properties.items():
                df[key].append(value)
            if is_train:
                df["canonical"].append(Chem.MolToSmiles(mol))
    return df


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


def calculate_probabilities(train_counts, gen_counts):
    keys = np.union1d(train_counts[0], gen_counts[0])
    n1, n2 = sum(train_counts[1]), sum(gen_counts[1])
    d1 = dict(zip(train_counts[0], train_counts[1]))
    d2 = dict(zip(gen_counts[0], gen_counts[1]))

    p1 = [(d1[key] / n1) if key in d1 else 0 for key in keys]
    p2 = [(d2[key] / n2) if key in d2 else 0 for key in keys]

    return p1, p2


def process_chunk(smiles, max_chunk_size, gen_canonical=None, existing_df=None):
    chunk = []
    common_smiles = set() if gen_canonical is not None else None
    df = defaultdict(list) if existing_df is None else existing_df

    for i, smile in enumerate(smiles):
        chunk.append(smile)
        if (i + 1) % max_chunk_size == 0:
            df = process_smiles(df, chunk, is_train=bool(gen_canonical))
            chunk = []

        if gen_canonical is not None and smile in gen_canonical:
            common_smiles.add(smile)

    df = process_smiles(df, chunk, is_train=bool(gen_canonical))
    return (df, common_smiles) if gen_canonical is not None else df


def process_outcomes(train_df, gen_df, output_file, sampled_file):

    org_counts = np.unique(np.concatenate(train_df["elements"]), return_counts=True)
    org_murcko_counts = np.unique(train_df["murcko"], return_counts=True)
    gen_counts = np.unique(np.concatenate(gen_df["elements"]), return_counts=True)
    gen_murcko_counts = np.unique(gen_df["murcko"], return_counts=True)

    p1, p2 = calculate_probabilities(org_counts, gen_counts)
    p_m1, p_m2 = calculate_probabilities(org_murcko_counts, gen_murcko_counts)
    fcd = FCD(canonize=False)

    res = [
        ("% valid", len(gen_df["all_mols"]) / len(gen_df["smiles"])),
        ("% novel", len(gen_df["mols"]) / len(gen_df["canonical"])),
        (
            "% unique",
            len(set(gen_df["canonical"])) / len(gen_df["canonical"]),
        ),
        ("KL divergence, atoms", scipy.stats.entropy(p2, p1)),
        ("Jensen-Shannon distance, atoms", jensenshannon(p2, p1)),
        ("Wasserstein distance, atoms", wasserstein_distance(p2, p1)),
        (
            "Jensen-Shannon distance, MWs",
            continuous_JSD(gen_df["mws"], train_df["mws"]),
        ),
        (
            "Jensen-Shannon distance, logP",
            continuous_JSD(gen_df["logp"], train_df["logp"]),
        ),
        (
            "Jensen-Shannon distance, Bertz TC",
            continuous_JSD(gen_df["tcs"], train_df["tcs"]),
        ),
        (
            "Jensen-Shannon distance, QED",
            continuous_JSD(gen_df["qed"], train_df["qed"]),
        ),
        (
            "Jensen-Shannon distance, TPSA",
            continuous_JSD(gen_df["tpsa"], train_df["tpsa"]),
        ),
        ("Internal diversity", internal_diversity(gen_df["fps"])),
        (
            "External diversity",
            external_diversity(gen_df["fps"], train_df["fps"]),
        ),
        (
            "External nearest-neighbor Tc",
            external_nn(gen_df["fps"], train_df["fps"]),
        ),
        ("Internal nearest-neighbor Tc", internal_nn(gen_df["fps"])),
        (
            "Jensen-Shannon distance, # of rings",
            discrete_JSD(gen_df["rings1"], train_df["rings1"]),
        ),
        (
            "Jensen-Shannon distance, # of aliphatic rings",
            discrete_JSD(gen_df["rings2"], train_df["rings2"]),
        ),
        (
            "Jensen-Shannon distance, # of aromatic rings",
            discrete_JSD(gen_df["rings3"], train_df["rings3"]),
        ),
        (
            "Jensen-Shannon distance, SA score",
            continuous_JSD(gen_df["SA"], train_df["SA"]),
        ),
        (
            "Jensen-Shannon distance, NP score",
            continuous_JSD(gen_df["NP"], train_df["NP"]),
        ),
        (
            "Jensen-Shannon distance, % sp3 carbons",
            continuous_JSD(gen_df["sp3"], train_df["sp3"]),
        ),
        (
            "Jensen-Shannon distance, % rotatable bonds",
            continuous_JSD(gen_df["rot"], train_df["rot"]),
        ),
        (
            "Jensen-Shannon distance, % stereocenters",
            continuous_JSD(gen_df["stereo"], train_df["stereo"]),
        ),
        (
            "Jensen-Shannon distance, Murcko scaffolds",
            jensenshannon(p_m2, p_m1),
        ),
        (
            "Jensen-Shannon distance, hydrogen donors",
            discrete_JSD(gen_df["donors"], train_df["donors"]),
        ),
        (
            "Jensen-Shannon distance, hydrogen acceptors",
            discrete_JSD(gen_df["acceptors"], train_df["acceptors"]),
        ),
        (
            "Frechet ChemNet distance",
            fcd(gen_df["canonical"], train_df["canonical"]),
        ),
    ]

    res = pd.DataFrame(res, columns=["outcome", "value"])
    res.insert(0, "input_file", os.path.basename(sampled_file))
    res.to_csv(
        output_file,
        index=False,
        compression="gzip" if str(output_file).endswith(".gz") else None,
    )

    res.reset_index(inplace=True, drop=True)
    return res


def calculate_outcomes(
    train_file, sampled_file, output_file, max_orig_mols, seed, max_chunk_size=100
):
    set_seed(seed)
    gen_df = defaultdict(list)

    gen_smiles = read_file(
        sampled_file, max_lines=max_orig_mols, stream=True, smile_only=True
    )
    train_smiles = read_file(
        train_file, max_lines=max_orig_mols, stream=True, smile_only=True
    )

    # TODO: Can this be done in chunk? This will likely be larger of a file than train
    for smile in gen_smiles:
        gen_df["smiles"].append(smile)
        if (mol := clean_mol(smile)) is not None:
            gen_df["all_mols"].append(mol)
            gen_df["canonical"].append(Chem.MolToSmiles(mol))
    gen_canonical = set(gen_df["canonical"])

    train_df, common = process_chunk(
        train_smiles, max_chunk_size, gen_canonical=gen_canonical
    )
    novel_smiles = gen_canonical.difference(common)
    gen_df = process_chunk(novel_smiles, max_chunk_size, existing_df=gen_df)

    return process_outcomes(train_df, gen_df, output_file, sampled_file)


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
