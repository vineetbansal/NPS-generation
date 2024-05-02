import argparse
import os
import numpy as np
import pandas as pd
import scipy.stats
from fcd_torch import FCD
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.MolSurf import TPSA
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from rdkit import rdBase
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
from clm.functions import (
    read_file,
    seed_type,
    set_seed,
    clean_mol,
    compute_fingerprint,
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
logger = logging.getLogger(__name__)


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
    "canonical_smile": Chem.MolToSmiles,
    "elements": lambda mol: [atom.GetSymbol() for atom in mol.GetAtoms()],
    "mws": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tcs": BertzCT,
    "tpsa": TPSA,
    "qed": safe_qed,
    "rings1": Lipinski.RingCount,
    "rings2": Lipinski.NumAliphaticRings,
    "rings3": Lipinski.NumAromaticRings,
    "SA": safe_sascorer,
    "NP": lambda mol: npscorer.scoreMol(mol, fscore),
    "sp3": Lipinski.FractionCSP3,
    "rot": pct_rotatable_bonds,
    "stereo": pct_stereocenters,
    "murcko": lambda mol: MurckoScaffoldSmiles(mol=mol),
    "donors": Lipinski.NumHDonors,
    "acceptors": Lipinski.NumHAcceptors,
    "fps": compute_fingerprint,
}


def smile_properties_dataframe(input_file, max_smiles=None):
    data = []
    for i, smile in enumerate(
        read_file(
            input_file,
            smile_only=True,
            stream=True,
            max_lines=max_smiles,
            randomize=True,
        ),
        start=1,
    ):
        if (mol := clean_mol(smile, raise_error=False)) is not None:
            row = tuple(fun(mol) for fun in molecular_properties.values())
        else:
            row = tuple([None] * len(molecular_properties))

        data.append((smile,) + row)
        if i % 1_000 == 0:
            logger.info(f"Processed {i} SMILES")

    df = pd.DataFrame(data, columns=["smile"] + list(molecular_properties.keys()))
    return df


def calculate_probabilities(*dicts):
    merged_keys = set()
    for dictionary in dicts:
        merged_keys.update(dictionary.keys())

    return_values = []
    for dictionary in dicts:
        total = sum(dictionary.values())
        return_values.append([dictionary.get(k, 0) / total for k in merged_keys])

    return return_values


def get_dataframes(train_file, sampled_file):
    logger.info(f"Reading training smiles from {train_file}")
    train_df = smile_properties_dataframe(train_file)

    logger.info(f"Reading sample smiles from {sampled_file}")
    sample_smiles_df = smile_properties_dataframe(sampled_file)

    sample_smiles_df["is_valid"] = sample_smiles_df.apply(
        lambda row: row["canonical_smile"] is not None, axis=1
    )
    n_valid_smiles = sample_smiles_df["is_valid"].sum()
    logger.info(f"{n_valid_smiles} valid SMILES out of {len(sample_smiles_df)}")

    sample_smiles_df["is_novel"] = sample_smiles_df.apply(
        lambda row: row["smile"] not in train_df["smile"], axis=1
    )
    n_novel_smiles = sample_smiles_df["is_novel"].sum()
    logger.info(f"{n_novel_smiles} novel SMILES out of {len(sample_smiles_df)}")

    logger.info("Re-reading sample file to obtain bin/other information")
    sample_bin_df = pd.read_csv(sampled_file)
    logger.info("Merging bin information")
    sample_df = sample_smiles_df.merge(
        sample_bin_df, left_on="smile", right_on="smiles"
    )

    return train_df, sample_df


def calculate_outcomes_dataframe(sample_df, train_df):
    train_element_distribution = dict(
        zip(*np.unique(np.concatenate(train_df["elements"]), return_counts=True))
    )
    train_murcko_distribution = dict(
        zip(*np.unique(train_df["murcko"], return_counts=True))
    )

    out = []
    for bin, bin_df in sample_df.groupby("bin"):
        logger.info(f"Calculating outcomes for bin {bin}")
        bin_df = bin_df.reset_index(drop=True)
        element_distribution = dict(
            zip(
                *np.unique(
                    np.concatenate(bin_df["elements"].to_numpy()), return_counts=True
                )
            )
        )

        murcko_distribution = dict(
            zip(*np.unique(bin_df["murcko"].to_numpy(), return_counts=True))
        )

        p1, p2 = calculate_probabilities(
            train_element_distribution, element_distribution
        )
        p_m1, p_m2 = calculate_probabilities(
            train_murcko_distribution, murcko_distribution
        )
        fcd = FCD(canonize=False)
        out.append(
            {
                "bin": bin,
                "n_mols": len(bin_df),
                "% valid": len(bin_df[bin_df["is_valid"]]) / len(bin_df),
                "% novel": len(bin_df[bin_df["is_novel"]]) / len(bin_df),
                "% unique": len(bin_df["smile"].unique()) / len(bin_df),
                "KL divergence, atoms": scipy.stats.entropy(p2, p1),
                "Jensen-Shannon distance, atoms": jensenshannon(p2, p1),
                "Wasserstein distance, atoms": wasserstein_distance(p2, p1),
                "Jensen-Shannon distance, MWs": continuous_JSD(
                    bin_df["mws"], train_df["mws"]
                ),
                "Jensen-Shannon distance, logP": continuous_JSD(
                    bin_df["logp"], train_df["logp"]
                ),
                "Jensen-Shannon distance, Bertz TC": continuous_JSD(
                    bin_df["tcs"], train_df["tcs"]
                ),
                "Jensen-Shannon distance, QED": continuous_JSD(
                    bin_df["qed"], train_df["qed"]
                ),
                "Jensen-Shannon distance, TPSA": continuous_JSD(
                    bin_df["tpsa"], train_df["tpsa"]
                ),
                "Internal diversity": internal_diversity(bin_df["fps"].to_numpy()),
                "External diversity": external_diversity(
                    bin_df["fps"].to_numpy(), train_df["fps"].to_numpy()
                ),
                "Internal nearest-neighbor Tc": internal_nn(bin_df["fps"].to_numpy()),
                "External nearest-neighbor Tc": external_nn(
                    bin_df["fps"].to_numpy(), train_df["fps"].to_numpy()
                ),
                "Jensen-Shannon distance, # of rings": discrete_JSD(
                    bin_df["rings1"], train_df["rings1"]
                ),
                "Jensen-Shannon distance, # of aliphatic rings": discrete_JSD(
                    bin_df["rings2"], train_df["rings2"]
                ),
                "Jensen-Shannon distance, # of aromatic rings": discrete_JSD(
                    bin_df["rings3"], train_df["rings3"]
                ),
                "Jensen-Shannon distance, SA score": continuous_JSD(
                    bin_df["SA"], train_df["SA"]
                ),
                "Jensen-Shannon distance, NP score": continuous_JSD(
                    bin_df["NP"], train_df["NP"]
                ),
                "Jensen-Shannon distance, % sp3 carbons": continuous_JSD(
                    bin_df["sp3"], train_df["sp3"]
                ),
                "Jensen-Shannon distance, % rotatable bonds": continuous_JSD(
                    bin_df["rot"], train_df["rot"]
                ),
                "Jensen-Shannon distance, % stereocenters": continuous_JSD(
                    bin_df["stereo"], train_df["stereo"]
                ),
                "Jensen-Shannon distance, Murcko scaffolds": jensenshannon(p_m2, p_m1),
                "Jensen-Shannon distance, hydrogen donors": discrete_JSD(
                    bin_df["donors"], train_df["donors"]
                ),
                "Jensen-Shannon distance, hydrogen acceptors": discrete_JSD(
                    bin_df["acceptors"], train_df["acceptors"]
                ),
                "Frechet ChemNet distance": fcd(
                    bin_df[bin_df["is_novel"]]["canonical_smile"],
                    train_df["canonical_smile"],
                ),
            }
        )
    out = pd.DataFrame(out)

    # Have 'bin' as a column and each of our other columns as rows in an
    # 'outcome' column, with values in a 'value' column
    out = out.melt(id_vars=["bin"], var_name="outcome", value_name="value")
    return out


def calculate_outcomes(sampled_file, train_file, output_file, seed=None):
    set_seed(seed)
    train_df, sample_df = get_dataframes(train_file, sampled_file)

    logger.info("Calculating outcomes")
    out = calculate_outcomes_dataframe(sample_df, train_df)

    # `input_file` column added for legacy reasons
    out["input_file"] = os.path.basename(sampled_file)
    out.to_csv(output_file, index=False)

    return out


def main(args):
    calculate_outcomes(
        train_file=args.train_file,
        sampled_file=args.sampled_file,
        output_file=args.output_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
