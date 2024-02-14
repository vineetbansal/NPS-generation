import argparse
import numpy as np
import os
import pandas as pd
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from functions import clean_mol, clean_mols, read_smiles
from NPS_generation.functions import set_seed, seed_type

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')


def add_args(parser):
    parser.add_argument('--ranks_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--pubchem_file', type=str)
    parser.add_argument('--sample_file', type=str)
    parser.add_argument('--err_ppm', type=int)
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--seed', type=seed_type, default=None, nargs="?", help="Random seed")
    return parser


def inner_write_formula_prior_CV(ranks_file, train_file, test_file, pubchem_file, sample_file, err_ppm, chunk_size,
                                 seed):
    set_seed(args.seed)

    # read training and test sets
    all_train_smiles = read_smiles(train_file)

    train_masses = []
    train_fmlas = []
    with tqdm(total=len(all_train_smiles)) as pbar:
        for i in range(0, len(all_train_smiles), chunk_size):
            smiles = all_train_smiles[i:i + chunk_size]
            mols = clean_mols(smiles, disable_progress=True)
            train_masses.extend([round(Descriptors.ExactMolWt(mol), 4) for mol in mols])
            train_fmlas.extend([rdMolDescriptors.CalcMolFormula(mol) for mol in mols])
            pbar.update(len(smiles))

    train = pd.DataFrame({'smiles': all_train_smiles,
                          'mass': train_masses,
                          'formula': train_fmlas})

    all_test_smiles = read_smiles(test_file)

    test_masses = []
    test_fmlas = []
    with tqdm(total=len(all_test_smiles)) as pbar:
        for i in range(0, len(all_train_smiles), chunk_size):
            smiles = all_test_smiles[i:i + chunk_size]
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
    pubchem = pd.read_csv(pubchem_file, delimiter='\t', header=None,
                          names=['smiles', 'mass', 'formula'])
    # assign frequencies as NAs
    pubchem = pubchem.assign(size=np.nan)

    print('Reading sample file from generative model')
    gen = pd.read_csv(sample_file)

    # set up outputs
    rank_df = pd.DataFrame()

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

            # compute 10 ppm range
            min_mass = (-err_ppm / 1e6 * query_mass) + query_mass
            max_mass = (err_ppm / 1e6 * query_mass) + query_mass

            # get matches to mass
            if key != "model":
                # get matches to mass and shuffle them to random order
                matches = query[query['mass'].between(min_mass, max_mass)]. \
                    sample(frac=1)
                matches = matches.assign(rank=np.arange(matches.shape[0]))
            else:
                # generative model: sort descending by mass
                matches = query[query['mass'].between(min_mass, max_mass)]. \
                    sort_values('size', ascending=False)
                matches = matches.assign(rank=np.arange(matches.shape[0]))

            # add 'target_' to all columns in matches
            matches.columns = 'target_' + matches.columns

            # get the rank of the correct formula
            rank = matches[matches['target_formula'] ==
                           row.formula][['target_size', 'target_rank',
                                         'target_source']]
            if rank.shape[0] > 1:
                rank = rank.head(1)
            elif rank.shape[0] == 0:
                rank = pd.DataFrame({'target_size': np.nan,
                                     'target_rank': np.nan,
                                     'target_source': key}, index=[0])

            # assign number of candidates
            n_formulas = len(matches.target_formula.unique())
            rank = rank.assign(n_candidates=n_formulas)

            # now, append rank to the growing data frame
            rank_row = pd.concat([pd.DataFrame([row]).reset_index(drop=True),
                                  rank.reset_index(drop=True)], axis=1)
            rank_df = pd.concat([rank_df, rank_row])

    # write to output files
    os.makedirs(os.path.dirname(ranks_file), exist_ok=True)
    rank_df.to_csv(ranks_file, index=False,
                  compression='gzip' if ranks_file.endswith('.gz') else None)

def main(args):
    inner_write_formula_prior_CV(ranks_file=args.ranks_file,
                                 train_file=args.train_file,
                                 test_file=args.test_file,
                                 pubchem_file=args.pubchem_file,
                                 sample_file=args.sample_file,
                                 err_ppm=args.err_ppm,
                                 chunk_size=args.chunk_size,
                                 seed=args.seed
                                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
