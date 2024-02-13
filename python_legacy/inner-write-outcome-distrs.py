"""
Calculate a set of outcomes for a list of SMILES, writing the complete
distribution and not just a summary statistic.
"""

import argparse
import os
import numpy as np
import pandas as pd
import random
import sys
from itertools import chain
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
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
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)

# import functions
from functions import clean_mols, read_smiles, pct_rotatable_bonds, \
    pct_stereocenters

### dynamically build CLI
parser = argparse.ArgumentParser()
## build the CLI
grid_file = git_dir + '/sh/grids/write-outcome-distrs.txt'
grid = pd.read_csv(grid_file, sep='\t')
for arg_name in list(grid):
    param_name = '--' + arg_name
    param_dtype = str(grid[arg_name].dtype)
    # convert to pandas
    param_type = {'object': str,
                  'int64': int,
                  'float64': float,
                  'bool': bool
                  }[param_dtype]
    parser.add_argument(param_name, type=param_type)

# parse all arguments
args = parser.parse_args()
print(args)

# create results container
res = pd.DataFrame()

# read SMILES
with open(args.input_file) as f:
    first_line = f.readline()

if ',' in first_line:
    if 'selfies' in first_line:
        sample = pd.read_csv(args.input_file)
        smiles = sample['selfies'].tolist()
    elif 'smiles' in first_line:
        sample = pd.read_csv(args.input_file)
        smiles = sample['smiles'].tolist()
    else:
        sample = pd.read_csv(args.input_file, names=['loss', 'smiles'])
        smiles = sample['smiles'].tolist()
else:
    smiles = read_smiles(args.input_file)

# optionally, subsample to a more tractable number of molecules
if len(smiles) > args.max_sampled_mols:
    random.seed(args.max_sampled_mols)
    smiles = random.sample(smiles, int(args.max_sampled_mols))

# convert SMILES to molecules
mols = clean_mols(smiles, representation=args.representation)
idxs = [idx for idx, mol in enumerate(mols) if mol]
mols = [mols[idx] for idx in idxs]
smiles = [smiles[idx] for idx in idxs]

# calculate descriptors
## heteroatom distribution
elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in mols]
counts = np.unique(list(chain(*elements)), return_counts=True)
# Murcko scaffolds
murcko = [MurckoScaffoldSmiles(mol=mol) for mol in mols]
murcko_counts = np.unique(murcko, return_counts=True)

## molecular weights
mws = [Descriptors.MolWt(mol) for mol in mols]
## logP
logp = [Descriptors.MolLogP(mol) for mol in tqdm(mols)]
## Bertz TC
tcs = [BertzCT(mol) for mol in tqdm(mols)]
## TPSA
tpsa = [Descriptors.TPSA(mol) for mol in mols]
## QED
qed = []
for mol in tqdm(mols):
    try:
        qed.append(Descriptors.qed(mol))
    except OverflowError:
        pass

## % of sp3 carbons
pct_sp3 = [Lipinski.FractionCSP3(mol) for mol in tqdm(mols)]
## % rotatable bonds
pct_rot = [pct_rotatable_bonds(mol) for mol in mols]
## % of stereocentres
pct_stereo = [pct_stereocenters(mol) for mol in mols]
## % heteroatoms
pct_hetero = [Lipinski.NumHeteroatoms(mol) / mol.GetNumAtoms() for mol in \
              tqdm(mols)]
## number of rings
rings = [Lipinski.RingCount(mol) for mol in tqdm(mols)]
ali_rings = [Lipinski.NumAliphaticRings(mol) for mol in tqdm(mols)]
aro_rings = [Lipinski.NumAromaticRings(mol) for mol in tqdm(mols)]
## hydrogen donors/acceptors
h_donors = [Lipinski.NumHDonors(mol) for mol in mols]
h_acceptors = [Lipinski.NumHAcceptors(mol) for mol in mols]
## SA score
SA = []
for mol in tqdm(mols):
    try:
        SA.append(sascorer.calculateScore(mol))
    except (OverflowError, ZeroDivisionError):
        pass

## NP-likeness
fscore = npscorer.readNPModel()
NP = [npscorer.scoreMol(mol, fscore) for mol in tqdm(mols)]

# add all outcomes to data frame
res = res.append(pd.DataFrame({'outcome': 'Molecular weight',
                               'value': mws }))
res = res.append(pd.DataFrame({'outcome': 'LogP',
                               'value': logp }))
res = res.append(pd.DataFrame({'outcome': 'BertzTC',
                               'value': tcs }))
res = res.append(pd.DataFrame({'outcome': 'TPSA',
                               'value': tpsa }))
res = res.append(pd.DataFrame({'outcome': 'QED',
                               'value': qed }))
res = res.append(pd.DataFrame({'outcome': '% sp3 carbons',
                               'value': pct_sp3 }))
res = res.append(pd.DataFrame({'outcome': '% rotatable bonds',
                               'value': pct_rot }))
res = res.append(pd.DataFrame({'outcome': '% stereocentres',
                               'value': pct_stereo }))
res = res.append(pd.DataFrame({'outcome': '% heteroatoms',
                               'value': pct_hetero}))
res = res.append(pd.DataFrame({'outcome': '# of rings',
                               'value': rings }))
res = res.append(pd.DataFrame({'outcome': '# of aliphatic rings',
                               'value': ali_rings }))
res = res.append(pd.DataFrame({'outcome': '# of aromatic rings',
                               'value': aro_rings }))
res = res.append(pd.DataFrame({'outcome': '# of hydrogen donors',
                               'value': h_donors }))
res = res.append(pd.DataFrame({'outcome': '# of hydrogen acceptors',
                               'value': h_acceptors }))
res = res.append(pd.DataFrame({'outcome': 'Synthetic accessibility score',
                               'value': SA }))
res = res.append(pd.DataFrame({'outcome': 'Natural product-likeness score',
                               'value': NP }))
for idx, element in enumerate(counts[0]):
    atom_count = counts[1][idx]
    res = res.append(pd.DataFrame({'outcome': '# atoms, ' + element,
                                   'value': [atom_count] }))
for idx, scaffold in enumerate(murcko_counts[0]):
    murcko_count = murcko_counts[1][idx]
    res = res.append(pd.DataFrame({'outcome': 'Murcko scaffold: ' + scaffold,
                                   'value': [murcko_count] }))

# make output directories
output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# write output
res.to_csv(args.output_file, index=False, compression='gzip')
