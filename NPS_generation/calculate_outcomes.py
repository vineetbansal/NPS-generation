
"""
Calculate a set of outcomes summarizing the quality of a set of generated
molecules.
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
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.AllChem import CalcNumAtomStereoCenters
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon, cosine
from tqdm import tqdm

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
# import from rdkit.Contrib module
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import npscorer

# set working directory
git_dir = os.path.expanduser("~/git/NPS_generation")
python_dir = git_dir + "/python"
os.chdir(python_dir)

# import functions
from functions import clean_mols, read_smiles, \
    continuous_KL, discrete_KL, \
    continuous_JSD, discrete_JSD, \
    continuous_EMD, discrete_EMD, \
    internal_diversity, external_diversity, \
    internal_nn, external_nn, \
    get_ecfp6_fingerprints

### CLI
parser = argparse.ArgumentParser(
        description='Quantity the performance of a generative language model')
parser.add_argument('--original_file', type=str,
                    help='file containing training SMILES')
parser.add_argument('--output_dir', type=str,
                    help='directory to save output to')
parser.add_argument('--stop_if_exists', dest='stop_if_exists',
                    action='store_true')
parser.add_argument('--minimal', dest='minimal',
                    help='calculate only % valid, % novel, and % uniques',
                    action='store_true')
parser.add_argument('--selfies', dest='selfies',
                    help='calculate outcomes for molecules in SELFIES format',
                    action='store_true')
parser.add_argument('--deepsmiles', dest='deepsmiles',
                    help='calculate outcomes for molecules in DeepSMILES format',
                    action='store_true')
parser.add_argument('--sampled_files', type=str, nargs='*',
                    help='file(s) containing sampled SMILES')
parser.set_defaults(stop_if_exists=False)
args = parser.parse_args()
print(args)

# make output directories
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# read the training set SMILES, and convert to moelcules
org_smiles = read_smiles(args.original_file)
org_mols = [mol for mol in clean_mols(org_smiles, selfies=args.selfies,
                                      deepsmiles=args.deepsmiles) if mol]
org_canonical = [Chem.MolToSmiles(mol) for mol in org_mols]

# define helper function to get # of rotatable bonds
def pct_rotatable_bonds(mol):
    n_bonds = mol.GetNumBonds()
    if n_bonds > 0:
        rot_bonds = Lipinski.NumRotatableBonds(mol) / n_bonds
    else:
        rot_bonds = 0
    return rot_bonds

# define helper function to get % of stereocenters
def pct_stereocentres(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms > 0:
        Chem.AssignStereochemistry(mol)
        pct_stereo = CalcNumAtomStereoCenters(mol) / n_atoms
    else:
        pct_stereo = 0
    return pct_stereo

# calculate training set descriptors
if not args.minimal:
    ## heteroatom distribution
    org_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for \
                     mol in org_mols]
    org_counts = np.unique(list(chain(*org_elements)), return_counts=True)
    ## molecular weights
    org_mws = [Descriptors.MolWt(mol) for mol in org_mols]
    ## logP
    org_logp = [Descriptors.MolLogP(mol) for mol in tqdm(org_mols)]
    ## Bertz TC
    org_tcs = [BertzCT(mol) for mol in tqdm(org_mols)]
    ## TPSA
    org_tpsa = [Descriptors.TPSA(mol) for mol in org_mols]
    ## QED
    org_qed = []
    for mol in org_mols:
        try:
            org_qed.append(Descriptors.qed(mol))
        except OverflowError:
            pass

    ## number of rings
    org_rings1 = [Lipinski.RingCount(mol) for mol in tqdm(org_mols)]
    org_rings2 = [Lipinski.NumAliphaticRings(mol) for mol in tqdm(org_mols)]
    org_rings3 = [Lipinski.NumAromaticRings(mol) for mol in tqdm(org_mols)]
    ## SA score
    org_SA = []
    for mol in tqdm(org_mols):
        try:
            org_SA.append(sascorer.calculateScore(mol))
        except (OverflowError, ZeroDivisionError):
            pass

    ## NP-likeness
    fscore = npscorer.readNPModel()
    org_NP = [npscorer.scoreMol(mol, fscore) for mol in tqdm(org_mols)]
    ## % sp3 carbons
    org_sp3 = [Lipinski.FractionCSP3(mol) for mol in org_mols]
    ## % rotatable bonds
    org_rot = [pct_rotatable_bonds(mol) for mol in org_mols]
    ## % of stereocentres
    org_stereo = [pct_stereocentres(mol) for mol in org_mols]
    # Murcko scaffolds
    org_murcko = []
    for mol in org_mols:
        try:
            org_murcko.append(MurckoScaffoldSmiles(mol=mol))
        except ValueError:
            pass
    # org_murcko = [MurckoScaffoldSmiles(mol=mol) for mol in org_mols]
    org_murcko_counts = np.unique(org_murcko, return_counts=True)
    ## hydrogen donors/acceptors
    org_donors = [Lipinski.NumHDonors(mol) for mol in org_mols]
    org_acceptors = [Lipinski.NumHAcceptors(mol) for mol in org_mols]

# loop over sampled files
for sampled_file in args.sampled_files:
    print("processing sampled SMILES file: " + str(sampled_file))

    # set up output
    sampled_filename = os.path.basename(sampled_file)
    output_filename = os.path.splitext(sampled_filename)[0] + \
        '-outcomes.csv.gz'
    output_file = os.path.join(args.output_dir, output_filename)

    # check if output file already exists
    if os.path.isfile(output_file) and args.stop_if_exists:
        print("  output file " + output_file + " exists: continuing...")
    else:
        # read generated SMILES and convert to molecules
        gen_smiles = read_smiles(sampled_file)
        gen_mols = [mol for mol in clean_mols(gen_smiles,
                                              selfies=args.selfies,
                                              deepsmiles=args.deepsmiles) if mol]
        gen_canonical = [Chem.MolToSmiles(mol) for mol in gen_mols]

        # create results container
        res = pd.DataFrame()

        # calculate descriptors
        ## outcome 1: % valid
        pct_valid = len(gen_mols) / len(gen_smiles)
        res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': '% valid',
                'value': [pct_valid] }))

        ## outcome 2: % novel
        # convert back to canonical SMILES for text-based comparison
        pct_novel = len([sm for sm in gen_canonical if not sm in \
                         org_canonical]) / len(gen_canonical)
        res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': '% novel',
                'value': [pct_novel] }))

        ## outcome 3: % unique
        pct_unique = len(set(gen_canonical)) / len(gen_canonical)
        res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': '% unique',
                'value': [pct_unique] }))

        if not args.minimal:
            ## outcome 4: K-L divergence of heteroatom distributions
            gen_elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for \
                             mol in gen_mols]
            gen_counts = np.unique(list(chain(*gen_elements)),
                                   return_counts=True)
            # get all unique keys
            keys = np.union1d(org_counts[0], gen_counts[0])
            n1, n2 = sum(org_counts[1]), sum(gen_counts[1])
            d1 = dict(zip(org_counts[0], org_counts[1]))
            d2 = dict(zip(gen_counts[0], gen_counts[1]))
            p1 = [d1[key] / n1 if key in d1.keys() else 0 for key in keys]
            p2 = [d2[key] / n2 if key in d2.keys() else 0 for key in keys]
            kl_atoms = scipy.stats.entropy([p + 1e-10 for p in p2],
                                           [p + 1e-10 for p in p1])
            jsd_atoms = jensenshannon(p2, p1)
            emd_atoms = wasserstein_distance(p2, p1)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, atoms',
                                'Jensen-Shannon distance, atoms',
                                'Wasserstein distance, atoms'],
                    'value': [kl_atoms, jsd_atoms, emd_atoms] }))

            ## outcome 5: K-L divergence of molecular weight
            gen_mws = [Descriptors.MolWt(mol) for mol in gen_mols]
            kl_mws = continuous_KL(gen_mws, org_mws)
            jsd_mws = continuous_JSD(gen_mws, org_mws)
            emd_mws = continuous_EMD(gen_mws, org_mws)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, MWs',
                                'Jensen-Shannon distance, MWs',
                                'Wasserstein distance, MWs'],
                    'value': [kl_mws, jsd_mws, emd_mws] }))

            ## outcome 6: K-L divergence of LogP
            gen_logp = [Descriptors.MolLogP(mol) for mol in gen_mols]
            kl_logp = continuous_KL(gen_logp, org_logp)
            jsd_logp = continuous_JSD(gen_logp, org_logp)
            emd_logp = continuous_EMD(gen_logp, org_logp)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, logP',
                                'Jensen-Shannon distance, logP',
                                'Wasserstein distance, logP'],
                    'value': [kl_logp, jsd_logp, emd_logp] }))

            ## outcome 7: K-L divergence of Bertz topological complexity
            gen_tcs = [BertzCT(mol) for mol in gen_mols]
            kl_tc = continuous_KL(gen_tcs, org_tcs)
            jsd_tc = continuous_JSD(gen_tcs, org_tcs)
            emd_tc = continuous_EMD(gen_tcs, org_tcs)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, Bertz TC',
                                'Jensen-Shannon distance, Bertz TC',
                                'Wasserstein distance, Bertz TC'],
                    'value': [kl_tc, jsd_tc, emd_tc] }))

            ## outcome 8: K-L divergence of QED
            gen_qed = []
            for mol in gen_mols:
                try:
                    gen_qed.append(Descriptors.qed(mol))
                except OverflowError:
                    pass

            kl_qed = continuous_KL(gen_qed, org_qed)
            jsd_qed = continuous_JSD(gen_qed, org_qed)
            emd_qed = continuous_EMD(gen_qed, org_qed)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, QED',
                                'Jensen-Shannon distance, QED',
                                'Wasserstein distance, QED'],
                    'value': [kl_qed, jsd_qed, emd_qed] }))

            ## outcome 9: K-L divergence of TPSA
            gen_tpsa = [Descriptors.TPSA(mol) for mol in gen_mols]
            kl_tpsa = continuous_KL(gen_tpsa, org_tpsa)
            jsd_tpsa = continuous_JSD(gen_tpsa, org_tpsa)
            emd_tpsa = continuous_EMD(gen_tpsa, org_tpsa)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, TPSA',
                                'Jensen-Shannon distance, TPSA',
                                'Wasserstein distance, TPSA'],
                    'value': [kl_tpsa, jsd_tpsa, emd_tpsa] }))

            ## outcome 10: internal diversity
            gen_fps = get_ecfp6_fingerprints(gen_mols)
            internal_div = internal_diversity(gen_fps)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': 'Internal diversity',
                    'value': [internal_div] }))

            ## outcome 11: median Tc to original set
            org_fps = get_ecfp6_fingerprints(org_mols)
            external_div = external_diversity(gen_fps, org_fps)
            res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': 'External diversity',
                'value': [external_div] }))

            ## also, summarize using nearest-neighbor instead of mean
            internal_nn = internal_nn(gen_fps)
            external_nn = external_nn(gen_fps, org_fps)
            res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': ['External nearest-neighbor Tc',
                            'Internal nearest-neighbor Tc'],
                'value': [external_nn, internal_nn] }))

            ## outcome 12: K-L divergence of number of rings
            gen_rings1 = [Lipinski.RingCount(mol) for mol in gen_mols]
            gen_rings2 = [Lipinski.NumAliphaticRings(mol) for mol in gen_mols]
            gen_rings3 = [Lipinski.NumAliphaticRings(mol) for mol in gen_mols]
            kl_rings1 = discrete_KL(gen_rings1, org_rings1)
            kl_rings2 = discrete_KL(gen_rings2, org_rings2)
            kl_rings3 = discrete_KL(gen_rings3, org_rings3)
            jsd_rings1 = discrete_JSD(gen_rings1, org_rings1)
            jsd_rings2 = discrete_JSD(gen_rings2, org_rings2)
            jsd_rings3 = discrete_JSD(gen_rings3, org_rings3)
            emd_rings1 = discrete_EMD(gen_rings1, org_rings1)
            emd_rings2 = discrete_EMD(gen_rings2, org_rings2)
            emd_rings3 = discrete_EMD(gen_rings3, org_rings3)
            res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': ['KL divergence, # of rings',
                            'KL divergence, # of aliphatic rings',
                            'KL divergence, # of aromatic rings',
                            'Jensen-Shannon distance, # of rings',
                            'Jensen-Shannon distance, # of aliphatic rings',
                            'Jensen-Shannon distance, # of aromatic rings',
                            'Wasserstein distance, # of rings',
                            'Wasserstein distance, # of aliphatic rings',
                            'Wasserstein distance, # of aromatic rings'],
                'value': [kl_rings1, kl_rings2, kl_rings3,
                          jsd_rings1, jsd_rings2, jsd_rings3,
                          emd_rings1, emd_rings2, emd_rings3] }))

            ## outcome 13: K-L divergence of SA score
            gen_SA = []
            for mol in gen_mols:
                try:
                    gen_SA.append(sascorer.calculateScore(mol))
                except (OverflowError, ZeroDivisionError):
                    pass

            kl_SA = continuous_KL(gen_SA, org_SA)
            jsd_SA = continuous_JSD(gen_SA, org_SA)
            emd_SA = continuous_EMD(gen_SA, org_SA)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, SA score',
                                'Jensen-Shannon distance, SA score',
                                'Wasserstein distance, SA score'],
                    'value': [kl_SA, jsd_SA, emd_SA] }))

            ## outcome 14: K-L divergence of NP-likeness
            gen_NP = []
            for mol in gen_mols:
                try:
                    gen_NP.append(npscorer.scoreMol(mol, fscore))
                except (OverflowError, ZeroDivisionError):
                    pass

            kl_NP = continuous_KL(gen_NP, org_NP)
            jsd_NP = continuous_JSD(gen_NP, org_NP)
            emd_NP = continuous_EMD(gen_NP, org_NP)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, NP score',
                                'Jensen-Shannon distance, NP score',
                                'Wasserstein distance, NP score'],
                    'value': [kl_NP, jsd_NP, emd_NP] }))

            ## outcome 15: K-L divergence of % sp3 carbons
            gen_sp3 = [Lipinski.FractionCSP3(mol) for mol in gen_mols]
            kl_sp3 = continuous_KL(gen_sp3, org_sp3)
            jsd_sp3 = continuous_JSD(gen_sp3, org_sp3)
            emd_sp3 = continuous_EMD(gen_sp3, org_sp3)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, % sp3 carbons',
                                'Jensen-Shannon distance, % sp3 carbons',
                                'Wasserstein distance, % sp3 carbons'],
                    'value': [kl_sp3, jsd_sp3, emd_sp3] }))

            ## outcome 16: K-L divergence of % rotatable bonds
            gen_rot = [pct_rotatable_bonds(mol) for mol in gen_mols]
            kl_rot = continuous_KL(gen_rot, org_rot)
            jsd_rot = continuous_JSD(gen_rot, org_rot)
            emd_rot = continuous_EMD(gen_rot, org_rot)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, % rotatable bonds',
                                'Jensen-Shannon distance, % rotatable bonds',
                                'Wasserstein distance, % rotatable bonds'],
                    'value': [kl_rot, jsd_rot, emd_rot] }))

            ## outcome 17: K-L divergence of % stereocenters
            gen_stereo = [pct_stereocentres(mol) for mol in gen_mols]
            kl_stereo = continuous_KL(gen_stereo, org_stereo)
            jsd_stereo = continuous_JSD(gen_stereo, org_stereo)
            emd_stereo = continuous_EMD(gen_stereo, org_stereo)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, % stereocenters',
                                'Jensen-Shannon distance, % stereocenters',
                                'Wasserstein distance, % stereocenters'],
                    'value': [kl_stereo, jsd_stereo, emd_stereo] }))

            ## outcome 18: K-L divergence of Murcko scaffolds
            gen_murcko = []
            for mol in gen_mols:
                try:
                    gen_murcko.append(MurckoScaffoldSmiles(mol=mol))
                except ValueError:
                    pass
            # gen_murcko = [MurckoScaffoldSmiles(mol=mol) for mol in gen_mols]
            gen_murcko_counts = np.unique(gen_murcko, return_counts=True)
            # get all unique keys
            keys = np.union1d(org_murcko_counts[0], gen_murcko_counts[0])
            n1, n2 = sum(org_murcko_counts[1]), sum(gen_murcko_counts[1])
            d1 = dict(zip(org_murcko_counts[0], org_murcko_counts[1]))
            d2 = dict(zip(gen_murcko_counts[0], gen_murcko_counts[1]))
            p1 = [d1[key] / n1 if key in d1.keys() else 0 for key in keys]
            p2 = [d2[key] / n2 if key in d2.keys() else 0 for key in keys]
            kl_murcko = scipy.stats.entropy([p + 1e-10 for p in p2],
                                            [p + 1e-10 for p in p1])
            jsd_murcko = jensenshannon(p2, p1)
            emd_murcko = wasserstein_distance(p2, p1)
            cos_murcko = cosine(p2, p1)
            res = res.append(pd.DataFrame({
                    'input_file': sampled_file,
                    'outcome': ['KL divergence, Murcko scaffolds',
                                'Jensen-Shannon distance, Murcko scaffolds',
                                'Wasserstein distance, Murcko scaffolds',
                                'Cosine distance, Murcko scaffolds'],
                    'value': [kl_murcko, jsd_murcko, emd_murcko,
                              cos_murcko] }))

            ## outcome 19: K-L divergence of # of hydrogen donors/acceptors
            gen_donors = [Lipinski.NumHDonors(mol) for mol in gen_mols]
            gen_acceptors = [Lipinski.NumHAcceptors(mol) for mol in gen_mols]
            kl_donors = discrete_KL(gen_donors, org_donors)
            kl_acceptors = discrete_KL(gen_acceptors, org_acceptors)
            jsd_donors = discrete_JSD(gen_donors, org_donors)
            jsd_acceptors = discrete_JSD(gen_acceptors, org_acceptors)
            emd_donors = discrete_EMD(gen_donors, org_donors)
            emd_acceptors = discrete_EMD(gen_acceptors, org_acceptors)
            res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': ['KL divergence, hydrogen donors',
                            'KL divergence, hydrogen acceptors',
                            'Jensen-Shannon distance, hydrogen donors',
                            'Jensen-Shannon distance, hydrogen acceptors',
                            'Wasserstein distance, hydrogen donors',
                            'Wasserstein distance, hydrogen acceptors'],
                'value': [kl_donors, kl_acceptors,
                          jsd_donors, jsd_acceptors,
                          emd_donors, emd_acceptors] }))

            ## outcome 20: Frechet ChemNet distance
            fcd = FCD(canonize=False)
            fcd_calc = fcd(gen_canonical, org_canonical)
            res = res.append(pd.DataFrame({
                'input_file': sampled_file,
                'outcome': 'Frechet ChemNet distance',
                'value': [fcd_calc] }))

        # write output
        res.to_csv(output_file, index=False, compression='gzip')
