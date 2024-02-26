"""
Calculate outcomes for a generative model, with respect to the entire set of
training molecules.
"""

import argparse
import os
import numpy as np
import pandas as pd
import scipy.stats
import sys
from fcd_torch import FCD
from itertools import chain
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from rdkit import rdBase
from clm.functions import (
    clean_mols,
    read_smiles,
    continuous_JSD,
    discrete_JSD,
    internal_diversity,
    external_diversity,
    internal_nn,
    external_nn,
    get_rdkit_fingerprints,
    pct_rotatable_bonds,
    pct_stereocenters,
    seed_type,
    set_seed,
)

rdBase.DisableLog("rdApp.error")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
import npscorer  # noqa: E402


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


def calculate_outcomes(train_file, sampled_file, output_file, max_orig_mols, seed):
    set_seed(seed)

    org_smiles = read_smiles(train_file)

    # optionally, subsample to a more tractable number of molecules
    if len(org_smiles) > max_orig_mols:
        org_smiles = np.random.choice(org_smiles, max_orig_mols)

    # convert to moleculess
    org_mols = [mol for mol in clean_mols(org_smiles) if mol]
    org_canonical = [Chem.MolToSmiles(mol) for mol in org_mols]

    # calculate training set descriptors
    # heteroatom distribution
    org_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in org_mols]
    org_counts = np.unique(list(chain(*org_elements)), return_counts=True)
    # molecular weights
    org_mws = [Descriptors.MolWt(mol) for mol in org_mols]
    # logP
    org_logp = [Descriptors.MolLogP(mol) for mol in tqdm(org_mols)]
    # Bertz TC
    org_tcs = [BertzCT(mol) for mol in tqdm(org_mols)]
    # TPSA
    org_tpsa = [Descriptors.TPSA(mol) for mol in org_mols]
    # QED
    org_qed = []
    for mol in org_mols:
        try:
            org_qed.append(Descriptors.qed(mol))
        except OverflowError:
            pass

    # number of rings
    org_rings1 = [Lipinski.RingCount(mol) for mol in tqdm(org_mols)]
    org_rings2 = [Lipinski.NumAliphaticRings(mol) for mol in tqdm(org_mols)]
    org_rings3 = [Lipinski.NumAromaticRings(mol) for mol in tqdm(org_mols)]
    # SA score
    org_SA = []
    for mol in tqdm(org_mols):
        try:
            org_SA.append(sascorer.calculateScore(mol))
        except (OverflowError, ZeroDivisionError):
            pass

    # NP-likeness
    fscore = npscorer.readNPModel()
    org_NP = [npscorer.scoreMol(mol, fscore) for mol in tqdm(org_mols)]
    # % sp3 carbons
    org_sp3 = [Lipinski.FractionCSP3(mol) for mol in org_mols]
    # % rotatable bonds
    org_rot = [pct_rotatable_bonds(mol) for mol in org_mols]
    # % of stereocentres
    org_stereo = [pct_stereocenters(mol) for mol in org_mols]
    # Murcko scaffolds
    org_murcko = [MurckoScaffoldSmiles(mol=mol) for mol in org_mols]
    org_murcko_counts = np.unique(org_murcko, return_counts=True)
    # hydrogen donors/acceptors
    org_donors = [Lipinski.NumHDonors(mol) for mol in org_mols]
    org_acceptors = [Lipinski.NumHAcceptors(mol) for mol in org_mols]

    # fingerprints
    org_fps = get_rdkit_fingerprints(org_mols)

    # now, read sampled SMILES
    with open(sampled_file) as f:
        first_line = f.readline()

    if "," in first_line:
        sample = pd.read_csv(sampled_file)
        gen_smiles = sample["smiles"].tolist()
    else:
        gen_smiles = read_smiles(sampled_file)

    # convert to molecules
    gen_mols = [mol for mol in clean_mols(gen_smiles) if mol]
    gen_canonical = [Chem.MolToSmiles(mol) for mol in tqdm(gen_mols)]

    # create output data frame
    res = pd.DataFrame()

    # calculate descriptors
    # outcome 1: % valid
    pct_valid = len(gen_mols) / len(gen_smiles)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {"input_file": sampled_file, "outcome": "% valid", "value": [pct_valid]}
            ),
        ]
    )

    # outcome 2: % novel
    # convert back to canonical SMILES for text-based comparison
    pct_novel = len([sm for sm in gen_canonical if sm not in org_smiles]) / len(
        gen_canonical
    )
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {"input_file": sampled_file, "outcome": "% novel", "value": [pct_novel]}
            ),
        ]
    )

    # outcome 3: % unique
    pct_unique = len(set(gen_canonical)) / len(gen_canonical)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": "% unique",
                    "value": [pct_unique],
                }
            ),
        ]
    )

    # remove known molecules before calculating divergence metrics
    train_smiles = set(org_smiles)
    gen_mols = [
        gen_mols[idx] for idx, sm in enumerate(gen_canonical) if sm not in train_smiles
    ]

    # outcome 4: K-L divergence of heteroatom distributions
    gen_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in gen_mols]
    gen_counts = np.unique(list(chain(*gen_elements)), return_counts=True)
    # get all unique keys
    keys = np.union1d(org_counts[0], gen_counts[0])
    n1, n2 = sum(org_counts[1]), sum(gen_counts[1])
    d1 = dict(zip(org_counts[0], org_counts[1]))
    d2 = dict(zip(gen_counts[0], gen_counts[1]))
    p1 = [d1[key] / n1 if key in d1.keys() else 0 for key in keys]
    p2 = [d2[key] / n2 if key in d2.keys() else 0 for key in keys]
    kl_atoms = scipy.stats.entropy(p2, p1)
    jsd_atoms = jensenshannon(p2, p1)
    emd_atoms = wasserstein_distance(p2, p1)

    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": [
                        "KL divergence, atoms",
                        "Jensen-Shannon distance, atoms",
                        "Wasserstein distance, atoms",
                    ],
                    "value": [kl_atoms, jsd_atoms, emd_atoms],
                }
            ),
        ]
    )

    # outcome 5: K-L divergence of molecular weight
    gen_mws = [Descriptors.MolWt(mol) for mol in gen_mols]
    jsd_mws = continuous_JSD(gen_mws, org_mws)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, MWs"],
                    "value": [jsd_mws],
                },
                index=[0],
            ),
        ]
    )

    # outcome 6: K-L divergence of LogP
    gen_logp = [Descriptors.MolLogP(mol) for mol in gen_mols]
    jsd_logp = continuous_JSD(gen_logp, org_logp)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, logP"],
                    "value": [jsd_logp],
                },
                index=[0],
            ),
        ]
    )

    # outcome 7: K-L divergence of Bertz topological complexity
    gen_tcs = [BertzCT(mol) for mol in gen_mols]
    jsd_tc = continuous_JSD(gen_tcs, org_tcs)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, Bertz TC"],
                    "value": [jsd_tc],
                },
                index=[0],
            ),
        ]
    )

    # outcome 8: K-L divergence of QED
    gen_qed = []
    for mol in gen_mols:
        try:
            gen_qed.append(Descriptors.qed(mol))
        except OverflowError:
            pass

    jsd_qed = continuous_JSD(gen_qed, org_qed)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, QED"],
                    "value": [jsd_qed],
                },
                index=[0],
            ),
        ]
    )

    # outcome 9: K-L divergence of TPSA
    gen_tpsa = [Descriptors.TPSA(mol) for mol in gen_mols]
    jsd_tpsa = continuous_JSD(gen_tpsa, org_tpsa)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, TPSA"],
                    "value": [jsd_tpsa],
                },
                index=[0],
            ),
        ]
    )

    # outcome 10: internal diversity
    gen_fps = get_rdkit_fingerprints(gen_mols)
    internal_div = internal_diversity(gen_fps)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": "Internal diversity",
                    "value": [internal_div],
                }
            ),
        ]
    )

    # outcome 11: median Tc to original set
    external_div = external_diversity(gen_fps, org_fps)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": "External diversity",
                    "value": [external_div],
                }
            ),
        ]
    )

    # also, summarize using nearest-neighbor instead of mean
    nn1 = internal_nn(gen_fps)
    nn2 = external_nn(gen_fps, org_fps)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": [
                        "External nearest-neighbor Tc",
                        "Internal nearest-neighbor Tc",
                    ],
                    "value": [nn1, nn2],
                }
            ),
        ]
    )

    # outcome 12: K-L divergence of number of rings
    gen_rings1 = [Lipinski.RingCount(mol) for mol in gen_mols]
    gen_rings2 = [Lipinski.NumAliphaticRings(mol) for mol in gen_mols]
    gen_rings3 = [Lipinski.NumAliphaticRings(mol) for mol in gen_mols]
    jsd_rings1 = discrete_JSD(gen_rings1, org_rings1)
    jsd_rings2 = discrete_JSD(gen_rings2, org_rings2)
    jsd_rings3 = discrete_JSD(gen_rings3, org_rings3)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": [
                        "Jensen-Shannon distance, # of rings",
                        "Jensen-Shannon distance, # of aliphatic rings",
                        "Jensen-Shannon distance, # of aromatic rings",
                    ],
                    "value": [jsd_rings1, jsd_rings2, jsd_rings3],
                }
            ),
        ]
    )

    # outcome 13: K-L divergence of SA score
    gen_SA = []
    for mol in gen_mols:
        try:
            gen_SA.append(sascorer.calculateScore(mol))
        except (OverflowError, ZeroDivisionError):
            pass

    jsd_SA = continuous_JSD(gen_SA, org_SA)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, SA score"],
                    "value": [jsd_SA],
                },
                index=[0],
            ),
        ]
    )

    # outcome 14: K-L divergence of NP-likeness
    gen_NP = []
    for mol in gen_mols:
        try:
            gen_NP.append(npscorer.scoreMol(mol, fscore))
        except (OverflowError, ZeroDivisionError):
            pass

    jsd_NP = continuous_JSD(gen_NP, org_NP)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, NP score"],
                    "value": [jsd_NP],
                },
                index=[0],
            ),
        ]
    )

    # outcome 15: K-L divergence of % sp3 carbons
    gen_sp3 = [Lipinski.FractionCSP3(mol) for mol in gen_mols]
    jsd_sp3 = continuous_JSD(gen_sp3, org_sp3)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, % sp3 carbons"],
                    "value": [jsd_sp3],
                },
                index=[0],
            ),
        ]
    )

    # outcome 16: K-L divergence of % rotatable bonds
    gen_rot = [pct_rotatable_bonds(mol) for mol in gen_mols]
    jsd_rot = continuous_JSD(gen_rot, org_rot)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, % rotatable bonds"],
                    "value": [jsd_rot],
                },
                index=[0],
            ),
        ]
    )

    # outcome 17: K-L divergence of % stereocenters
    gen_stereo = [pct_stereocenters(mol) for mol in gen_mols]
    jsd_stereo = continuous_JSD(gen_stereo, org_stereo)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, % stereocenters"],
                    "value": [jsd_stereo],
                },
                index=[0],
            ),
        ]
    )

    # outcome 18: K-L divergence of Murcko scaffolds
    gen_murcko = [MurckoScaffoldSmiles(mol=mol) for mol in gen_mols]
    gen_murcko_counts = np.unique(gen_murcko, return_counts=True)
    # get all unique keys
    keys = np.union1d(org_murcko_counts[0], gen_murcko_counts[0])
    n1, n2 = sum(org_murcko_counts[1]), sum(gen_murcko_counts[1])
    d1 = dict(zip(org_murcko_counts[0], org_murcko_counts[1]))
    d2 = dict(zip(gen_murcko_counts[0], gen_murcko_counts[1]))
    p1 = [d1[key] / n1 if key in d1.keys() else 0 for key in keys]
    p2 = [d2[key] / n2 if key in d2.keys() else 0 for key in keys]
    jsd_murcko = jensenshannon(p2, p1)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": ["Jensen-Shannon distance, Murcko scaffolds"],
                    "value": jsd_murcko,
                },
                index=[0],
            ),
        ]
    )

    # outcome 19: K-L divergence of # of hydrogen donors/acceptors
    gen_donors = [Lipinski.NumHDonors(mol) for mol in gen_mols]
    gen_acceptors = [Lipinski.NumHAcceptors(mol) for mol in gen_mols]
    jsd_donors = discrete_JSD(gen_donors, org_donors)
    jsd_acceptors = discrete_JSD(gen_acceptors, org_acceptors)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": [
                        "Jensen-Shannon distance, hydrogen donors",
                        "Jensen-Shannon distance, hydrogen acceptors",
                    ],
                    "value": [jsd_donors, jsd_acceptors],
                }
            ),
        ]
    )

    # outcome 20: Frechet ChemNet distance
    fcd = FCD(canonize=False)
    fcd_calc = fcd(gen_canonical, org_canonical)
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "input_file": sampled_file,
                    "outcome": "Frechet ChemNet distance",
                    "value": [fcd_calc],
                }
            ),
        ]
    )

    res.to_csv(
        output_file,
        index=False,
        compression="gzip" if str(output_file).endswith(".gz") else None,
    )


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
