import argparse
import logging
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from clm.functions import (
    read_file,
    write_smiles,
    clean_mols,
    remove_salts_solvents,
    NeutraliseCharges,
    get_inchikey,
)
from clm.datasets import Vocabulary

logger = logging.getLogger(__name__)
VALID_ELEMENTS = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"]


def add_args(parser):
    parser.add_argument(
        "--input-file", type=str, required=True, help="File path of smiles file"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Output smiles file path"
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=3,
        help="Minimum number of heavy atoms that a valid molecule should have",
    )
    parser.add_argument(
        "--no-neutralise",
        action="store_true",
        help="Do not neutralise charges (default False)",
    )
    parser.add_argument(
        "--valid-atoms",
        nargs="+",
        help="Elements for valid molecules (default %(default)s)",
        default=VALID_ELEMENTS,
    )
    parser.add_argument(
        "--remove-rare",
        action="store_true",
        help="Remove molecules with tokens found in <0.01% or <10 molecules",
    )
    parser.add_argument(
        "--max-input-smiles",
        type=int,
        default=None,
        help="Maximum input smiles to read (useful for testing)",
    )

    return parser


def preprocess(
    input_file,
    output_file,
    max_input_smiles=None,
    neutralise=True,
    min_heavy_atoms=3,
    valid_atoms=None,
    remove_rare=False,
    chunk_size=100000,
):
    logger.info("reading input SMILES ...")

    all_smiles = read_file(
        smiles_file=input_file,
        max_lines=max_input_smiles,
        smile_only=True,
    )

    def preprocess_chunk(
        input_smiles, neutralise=True, min_heavy_atoms=3, valid_atoms=None
    ):
        logger.info("Preprocessing chunk of {} SMILES".format(len(input_smiles)))
        logger.info(
            "converting {} input SMILES to molecules ...".format(len(input_smiles))
        )
        mols = clean_mols(input_smiles)

        logger.info("Removing heavy atoms from {} molecules ...".format(len(mols)))
        if min_heavy_atoms > 0:
            mols = [
                remove_salts_solvents(mol, hac=min_heavy_atoms) if mol else None
                for mol in mols
            ]

        if neutralise:
            logger.info("Neutralising charges from {} molecules ...".format(len(mols)))
            mols = [NeutraliseCharges(mol) if mol else None for mol in mols]

        elements = [
            [atom.GetSymbol() for atom in mol.GetAtoms()] if mol else None
            for mol in mols
        ]
        valid_atoms = valid_atoms or VALID_ELEMENTS
        valid = set(valid_atoms)
        for idx, atoms in enumerate(elements):
            if atoms is not None and len(set(atoms) - valid) > 0:
                mols[idx] = None

        mols = [mol for mol in mols if mol is not None]

        return mols

    mols = []
    with tqdm(total=len(all_smiles)) as pbar:
        for i in range(0, len(all_smiles), chunk_size):
            input_smiles = all_smiles[i : i + chunk_size]
            _mols = preprocess_chunk(
                input_smiles=input_smiles,
                neutralise=neutralise,
                min_heavy_atoms=min_heavy_atoms,
                valid_atoms=valid_atoms,
            )
            mols.extend(_mols)
            pbar.update(len(input_smiles))

    # InchI Keys are a reliable way to identify duplicates
    logger.info(f"{len(mols)} total molecules")
    inchikeys = np.array([get_inchikey(mol) for mol in mols])
    inchikeys, indices = np.unique(inchikeys, return_index=True)
    logger.info(f"{len(indices)} unique InChI keys found")

    logger.info(f"converting {len(indices)} molecules back to SMILES")
    smiles = [Chem.MolToSmiles(mol) for mol in np.array(mols)[indices]]
    smiles = [sm for sm in smiles if sm != ""]

    if remove_rare:
        logger.info("Creating vocabulary")
        vocabulary = Vocabulary(smiles=smiles)
        logger.info(f"Trimming vocabulary of size {len(vocabulary)}")
        n_smiles = len(smiles)
        for token in vocabulary.characters:
            token_smiles = [sm for sm in smiles if token in vocabulary.tokenize(sm)]
            pct_smiles = len(token_smiles) / n_smiles
            if pct_smiles < 0.01 / 100 or len(token_smiles) <= 10:
                smiles = np.setdiff1d(smiles, token_smiles)

    write_smiles(smiles, output_file, mode="w", add_inchikeys=True)


def main(args):
    preprocess(
        input_file=args.input_file,
        output_file=args.output_file,
        max_input_smiles=args.max_input_smiles,
        neutralise=not args.no_neutralise,
        min_heavy_atoms=args.min_heavy_atoms,
        valid_atoms=args.valid_atoms,
        remove_rare=args.remove_rare,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
