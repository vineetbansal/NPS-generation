"""
Take an input SMILES file, and augment it by some fixed factor via SMILES
enumeration.
"""

import argparse
import sys

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# import SmilesEnumerator
from NPS_generation.util.SmilesEnumerator import SmilesEnumerator
from NPS_generation.functions import read_smiles, write_smiles


def main(input_file=None, output_file=None, enum_factor=None):
    ### CLI
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--enum_factor', type=int,
                        help='factor to augment the dataset by')


    args = parser.parse_known_args()[0]

    if input_file is None:
        input_file = args.input_file
    if output_file is None:
        output_file = args.output_file
    if enum_factor is None:
        enum_factor = args.enum_factor


    # check output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)



    # read SMILES
    smiles = read_smiles(input_file)
    # convert to numpy array
    smiles = np.asarray(smiles)

    # create enumerator
    sme = SmilesEnumerator(canonical=False, enum=True)

    # also store and write information about enumerated library size
    summary = pd.DataFrame()

    # enumerate potential SMILES
    enum = []
    max_tries = 200  ## randomized SMILES to generate for each input structure
    for sm_idx, sm in enumerate(tqdm(smiles)):
        tries = []
        for try_idx in range(max_tries):
            this_try = sme.randomize_smiles(sm)
            tries.append(this_try)
            tries = [rnd for rnd in np.unique(tries)]
            if len(tries) > enum_factor:
                tries = tries[:enum_factor]
                break
        enum.extend(tries)

    # write to line-delimited file
    write_smiles(enum, output_file)
    print("wrote " + str(len(enum)) + " SMILES to output file: " + \
          output_file)

    return 0


if __name__ == "__main__":
    main()
