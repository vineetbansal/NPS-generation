import argparse
import numpy as np
import os
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.DataStructs import FingerprintSimilarity
from tqdm import tqdm

from functions import clean_mol, clean_mols, get_ecfp6_fingerprints, \
    read_smiles

# suppress rdkit errors
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

parser = argparse.ArgumentParser()

parser.add_argument('--ranks_file', type=str)
parser.add_argument('--tc_file', type=str)
parser.add_argument('--train_file', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--pubchem_file', type=str)
parser.add_argument('--sample_file', type=str)
parser.add_argument('--err_ppm', type=int)
parser.add_argument('--chunk_size', type=int, default=100000)

# parse all arguments
args = parser.parse_args()
print(args)

# read training and test sets
all_train_smiles = read_smiles(args.train_file)

train_masses = []
train_fmlas = []
with tqdm(total=len(all_train_smiles)) as pbar:
    for i in range(0, len(all_train_smiles), args.chunk_size):
        smiles = all_train_smiles[i:i+args.chunk_size]
        mols = clean_mols(smiles, disable_progress=True)
        train_masses.extend([round(Descriptors.ExactMolWt(mol), 4) for mol in mols])
        train_fmlas.extend([rdMolDescriptors.CalcMolFormula(mol) for mol in mols])
        pbar.update(len(smiles))

train = pd.DataFrame({'smiles': all_train_smiles,
                      'mass': train_masses,
                      'formula': train_fmlas})

all_test_smiles = read_smiles(args.test_file)

test_masses = []
test_fmlas = []
with tqdm(total=len(all_test_smiles)) as pbar:
    for i in range(0, len(all_train_smiles), args.chunk_size):
        smiles = all_test_smiles[i:i+args.chunk_size]
        mols = clean_mols(smiles, disable_progress=True)
        test_masses.extend([round(Descriptors.ExactMolWt(mol), 4) for mol in mols])
        test_fmlas.extend([rdMolDescriptors.CalcMolFormula(mol) for mol in mols])
        pbar.update(len(smiles))

test = pd.DataFrame({'smiles': all_test_smiles,
                      'mass': test_masses,
                      'formula': test_fmlas})
test = test.assign(mass_known=test['mass'].isin(train_masses))
test = test.assign(formula_known=test['formula'].isin(train_fmlas))

# assign frequencies as NAs
train = train.assign(size=np.nan)

print('Reading PubChem file')
pubchem = pd.read_csv(args.pubchem_file, delimiter='\t', header=None,
                      names=['smiles', 'mass', 'formula'])
# assign frequencies as NAs
pubchem = pubchem.assign(size=np.nan)

print('Reading sample file from generative model')
gen = pd.read_csv(args.sample_file)

# set up outputs
rank_df = pd.DataFrame()
tc_df = pd.DataFrame()

# iterate through PubChem vs. generative model
inputs = {'model': gen.assign(source='model'),
          'PubChem': pubchem.assign(source='PubChem'),
          'train': train.assign(source='train')}

# match on formula and mass
for key, query in inputs.items():
    print(f'Generating statistics for model {key}')
    for row in tqdm(test.itertuples(), total=test.shape[0]):
        # get formula and exact mass
        query_mol = clean_mol(row.smiles)
        query_mass = Descriptors.ExactMolWt(query_mol)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 3,
                                                         nBits=1024)
        
        # compute 10 ppm range
        min_mass = (-args.err_ppm / 1e6 * query_mass) + query_mass
        max_mass = (args.err_ppm / 1e6 * query_mass) + query_mass
        
        # get matches to mass
        if key != "model":
            # get matches to mass and shuffle them to random order
            matches = query[query['mass'].between(min_mass, max_mass)].\
                sample(frac=1)
            matches = matches.assign(rank=np.arange(matches.shape[0]))
        else:
            # generative model: sort descending by mass
            matches = query[query['mass'].between(min_mass, max_mass)].\
                sort_values('size', ascending=False)
            matches = matches.assign(rank=np.arange(matches.shape[0]))
        
        # add 'target_' to all columns in matches
        matches.columns = 'target_' + matches.columns
        
        # get the rank of the correct molecule
        rank = matches[matches['target_smiles'] ==
                       row.smiles][['target_size', 'target_rank',
                                    'target_source']]
        if rank.shape[0] > 1:
            rank = rank.head(1)
        elif rank.shape[0] == 0:
            rank = pd.DataFrame({'target_size': np.nan,
                                 'target_rank': np.nan,
                                 'target_source': key}, index=[0])
        
        # assign number of candidates
        rank = rank.assign(n_candidates=matches.shape[0])
        
        # get the Tc of the top match
        top_n = 1
        tc = matches
        if tc.shape[0] > top_n:
            tc = tc.head(top_n)
        # add a random match
        if key == "model" and matches.shape[0] > 1:
            rnd = matches.tail(-top_n).sample()
            tc = pd.concat([tc, rnd])
        
        # compute Tc
        if tc.shape[0] > 0:
            target_mols = clean_mols(tc['target_smiles'].values, disable_progress=True)
            keep = [idx for idx, mol in enumerate(target_mols) if mol]
            tc = tc.iloc[keep, :]
            target_mols = [mol for mol in target_mols if mol]
            target_fps = get_ecfp6_fingerprints(target_mols)
            tcs = [FingerprintSimilarity(query_fp, target_fp) for \
                   target_fp in target_fps]
            tc = tc.assign(Tc=tcs)
        else:
            # fill with empty data frame if there were no matches
            tc = pd.DataFrame({'target_size': np.nan,
                               'target_rank': np.nan,
                               'target_source': key,
                               'Tc': np.nan}, index=[0])
        
        # now, append Tc to the growing data frame
        tc_row = pd.concat(
            [pd.DataFrame([row]).iloc[np.full(tc.shape[0], 0)].\
             reset_index(drop=True), tc.reset_index(drop=True)], axis=1)
        tc_df = pd.concat([tc_df, tc_row])
        
        # do the same for rank
        rank_row = pd.concat([pd.DataFrame([row]).reset_index(drop=True),
                              rank.reset_index(drop=True)], axis=1)
        rank_df = pd.concat([rank_df, rank_row])

# write to output files
os.makedirs(os.path.dirname(args.ranks_file), exist_ok=True)
rank_df.to_csv(args.ranks_file, index=False,
               compression='gzip')
os.makedirs(os.path.dirname(args.tc_file), exist_ok=True)
tc_df.to_csv(args.tc_file, index=False,
             compression='gzip')
