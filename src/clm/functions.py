import deepsmiles
import numpy as np
import os
import os.path

import warnings
from selfies import decoder
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import torch

converter = deepsmiles.Converter(rings=True, branches=True)


def clean_mol(smiles, stereochem=False, selfies=False, deepsmiles=False):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
    """
    if selfies:
        selfies = smiles.replace("<PAD>", "[nop]")
        smiles = decoder(selfies)
    elif deepsmiles:
        deepsmiles = smiles
        try:
            smiles = converter.decode(deepsmiles)
        except ValueError:
            raise ValueError(f"invalid DeepSMILES: {deepsmiles}")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid SMILES: " + str(smiles))
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def clean_mols(all_smiles, stereochem=False, selfies=False, deepsmiles=False, disable_progress=False):
    """
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols = []
    for smiles in tqdm(all_smiles, disable=disable_progress):
        try:
            mol = clean_mol(smiles, stereochem, selfies, deepsmiles)
            mols.append(mol)
        except ValueError:
            mols.append(None)
    return mols


def remove_salts_solvents(mol, hac=3):
    """
    Remove solvents and ions have max 'hac' heavy atoms.
    This function was obtained from the mol2vec package,
    available at:
        https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py
    """
    # split molecule into fragments
    fragments = list(rdmolops.GetMolFrags(mol, asMols=True))
    # keep heaviest only
    # fragments.sort(reverse=True, key=lambda m: m.GetNumAtoms())
    # remove fragments with < 'hac' heavy atoms
    fragments = [fragment for fragment in fragments if fragment.GetNumAtoms() > hac]
    #
    if len(fragments) > 1:
        warnings.warn(
            "molecule contains >1 fragment with >" + str(hac) + " heavy atoms"
        )
        return None
    elif len(fragments) == 0:
        warnings.warn(
            "molecule contains no fragments with >" + str(hac) + " heavy atoms"
        )
        return None
    else:
        return fragments[0]



def get_ecfp6_fingerprints(mols, include_none=False):
    """
    Get ECFP6 fingerprints for a list of molecules. Optionally,
    handle `None` values by returning a `None` value in that
    position.
    """
    fps = []
    for mol in mols:
        if mol is None and include_none:
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
            fps.append(fp)
    return fps

def read_smiles(smiles_file, max_lines=None):
    """
    Read a list of SMILES from a line-delimited file.
    """
    smiles = []
    lines = 0
    with open(smiles_file, "r") as f:
        while line := f.readline().strip():
            smiles.append(line)
            lines += 1
            if max_lines != 0 and max_lines is not None and lines == max_lines:
                break
    return smiles


def write_smiles(smiles, smiles_file, mode="w"):
    """
    Write a list of SMILES to a line-delimited file.
    """
    os.makedirs(os.path.dirname(smiles_file), exist_ok=True)
    with open(smiles_file, mode) as f:
        for sm in smiles:
            _ = f.write(sm + "\n")



"""
rdkit contributed code to neutralize charged molecules;
obtained from:
    https://www.rdkit.org/docs/Cookbook.html
    http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg02669.html
"""


def _InitialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ("[n+;H]", "n"),
        # Amines
        ("[N+;!H0]", "N"),
        # Carboxylic acids and alcohols
        ("[$([O-]);!$([O-][#7])]", "O"),
        # Thiols
        ("[S-;X1]", "S"),
        # Sulfonamides
        ("[$([N-;X2]S(=O)=O)]", "N"),
        # Enamines
        ("[$([N-;X2][C,N]=C)]", "N"),
        # Tetrazoles
        ("[n-]", "[nH]"),
        # Sulfoxides
        ("[$([S-]=O)]", "S"),
        # Amides
        ("[$([N-]C=O)]", "N"),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


_reactions = None


def NeutraliseCharges(mol, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = _InitialiseNeutralisationReactions()
        reactions = _reactions
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def seed_type(value):
    # A "type" useful for argparse arguments for random seeds
    # that can come in as "None" (e.g. from a snakemake workflow)
    return None if value == "None" else int(value)
