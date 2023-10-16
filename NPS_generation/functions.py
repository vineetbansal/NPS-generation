import deepsmiles
import numpy as np
import os
import pandas as pd
import random
import re
import warnings
from selfies import decoder
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.DataStructs import FingerprintSimilarity
from scipy import histogram
from scipy.stats import entropy, gaussian_kde, wasserstein_distance
from scipy.spatial.distance import jensenshannon

converter = deepsmiles.Converter(rings=True, branches=True)

def clean_mol(smiles, stereochem=False, selfies=False, deepsmiles=False):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
    """
    if selfies:
        selfies = smiles
        smiles = decoder(selfies)
    elif deepsmiles:
        deepsmiles = smiles
        try:
            smiles = converter.decode(deepsmiles)
        except:
            raise ValueError("invalid DeepSMILES: " + str(deepsmiles))
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid SMILES: " + str(smiles))
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol

def clean_mols(all_smiles, stereochem=False, selfies=False, deepsmiles=False):
    """
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols = []
    for smiles in tqdm(all_smiles):
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
    fragments = list(rdmolops.GetMolFrags(mol, asMols = True))
    ## keep heaviest only
    ## fragments.sort(reverse=True, key=lambda m: m.GetNumAtoms())
    # remove fragments with < 'hac' heavy atoms
    fragments = [fragment for fragment in fragments if \
                 fragment.GetNumAtoms() > hac]
    #
    if len(fragments) > 1:
        warnings.warn("molecule contains >1 fragment with >" + str(hac) + \
                      " heavy atoms")
        return None
    elif len(fragments) == 0:
        warnings.warn("molecule contains no fragments with >" + str(hac) + \
                      " heavy atoms")
        return None
    else:
        return fragments[0]

def continuous_KL(generated_dist, original_dist, tol=1e-10):
    """
    Calculate the K-L divergence between two continuous distributions,
    using (Gaussian) kernel density estimation.

    Modified from the GuacaMol implementation at
    https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py

    Args:
        generated_dist: (list) distribution of real-valued continuous
          properties (e.g. logP) calculated from generated molecules
        original_dist:  (list) distribution of real-valued continuous
          properties (e.g. logP) calculated from training set molecules
    """
    gen_kde = gaussian_kde(generated_dist)
    org_kde = gaussian_kde(original_dist)
    vec = np.hstack([generated_dist, original_dist])
    x_eval = np.linspace(vec.min(), vec.max(), num=1000)
    P = gen_kde(x_eval) + tol
    Q = org_kde(x_eval) + tol
    return entropy(P, Q)

def continuous_JSD(generated_dist, original_dist, tol=1e-10):
    gen_kde = gaussian_kde(generated_dist)
    org_kde = gaussian_kde(original_dist)
    vec = np.hstack([generated_dist, original_dist])
    x_eval = np.linspace(vec.min(), vec.max(), num=1000)
    P = gen_kde(x_eval) + tol
    Q = org_kde(x_eval) + tol
    return jensenshannon(P, Q)

def continuous_EMD(generated_dist, original_dist, tol=1e-10):
    gen_kde = gaussian_kde(generated_dist)
    org_kde = gaussian_kde(original_dist)
    vec = np.hstack([generated_dist, original_dist])
    x_eval = np.linspace(vec.min(), vec.max(), num=1000)
    P = gen_kde(x_eval) + tol
    Q = org_kde(x_eval) + tol
    return wasserstein_distance(P, Q)

def discrete_KL(generated_dist, original_dist, tol=1e-10):
    """
    Calculate the K-L divergence between two discrete distributions.

    Modified from the GuacaMol implementation at
    https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py
    """
    gen, bins = histogram(generated_dist, density=True)
    org, bins = histogram(original_dist, density=True)
    gen += tol
    org += tol
    return entropy(gen, org)

def discrete_JSD(generated_dist, original_dist, tol=1e-10):
    gen, bins = histogram(generated_dist, density=True)
    org, bins = histogram(original_dist, density=True)
    gen += tol
    org += tol
    return jensenshannon(gen, org)

def discrete_EMD(generated_dist, original_dist, tol=1e-10):
    gen, bins = histogram(generated_dist, density=True)
    org, bins = histogram(original_dist, density=True)
    gen += tol
    org += tol
    return wasserstein_distance(gen, org)

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
    return(fps)

def internal_diversity(fps, sample_size=1e4, summarise=True):
    """
    Calculate the internal diversity, defined as the mean intra-set Tanimoto
    coefficient, between a set of fingerprints. For large sets, calculating the
    entire matrix is prohibitive, so a random set of molecules are sampled.
    """
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = random.randint(0, len(fps) - 1)
        idx2 = random.randint(0, len(fps) - 1)
        fp1 = fps[idx1]
        fp2 = fps[idx2]
        tcs.append(FingerprintSimilarity(fp1, fp2))
        counter += 1
    if summarise:
        return np.mean(tcs)
    else:
        return tcs

def external_diversity(fps1, fps2, sample_size=1e4, summarise=True):
    """
    Calculate the external diversity, defined as the mean inter-set Tanimoto
    coefficient, between two sets of fingerprints. For large sets, calculating
    the entire matrix is prohibitive, so a random set of molecules are sampled.
    """
    #
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = random.randint(0, len(fps1) - 1)
        idx2 = random.randint(0, len(fps2) - 1)
        fp1 = fps1[idx1]
        fp2 = fps2[idx2]
        tcs.append(FingerprintSimilarity(fp1, fp2))
        counter += 1
    if summarise:
        return np.mean(tcs)
    else:
        return tcs

def internal_nn(fps, sample_size=1e3, summarise=True):
    """
    Calculate the nearest-neighbor Tanimoto coefficient within a set of
    fingerprints.
    """
    counter = 0
    nns = []
    while counter < sample_size:
        idx1 = random.randint(0, len(fps) - 1)
        fp1 = fps[idx1]
        tcs = []
        for idx2 in range(len(fps)):
            if idx1 != idx2:
                fp2 = fps[idx2]
                tcs.append(FingerprintSimilarity(fp1, fp2))
        nn = np.max(tcs)
        nns.append(nn)
        counter += 1
    if summarise:
        return np.mean(nns)
    else:
        return nns

def external_nn(fps1, fps2, sample_size=1e3, summarise=True):
    """
    Calculate the nearest-neighbor Tanimoto coefficient, searching one set of
    fingerprints against a second set.
    """
    counter = 0
    nns = []
    while counter < sample_size:
        idx1 = random.randint(0, len(fps1) - 1)
        fp1 = fps1[idx1]
        tcs = []
        for idx2 in range(len(fps2)):
            fp2 = fps2[idx2]
            tcs.append(FingerprintSimilarity(fp1, fp2))
        nn = np.max(tcs)
        nns.append(nn)
        counter += 1
    if summarise:
        return np.mean(nns)
    else:
        return nns

def replace_halogen(smiles):
    """
    Replace halogens with single letters (Cl -> L and Br -> R), following
    Olivecrona et al. (J Cheminf 2018).
    """
    br = re.compile('Br')
    cl = re.compile('Cl')
    smiles = br.sub('R', smiles)
    smiles = cl.sub('L', smiles)
    return smiles

def read_smiles(smiles_file):
    """
    Read a list of SMILES from a line-delimited file.
    """
    smiles = []
    with open(smiles_file, 'r') as f:
        smiles.extend([line.strip() for line in f.readlines() \
                       if line.strip()])
    return smiles

def write_smiles(smiles, smiles_file):
    """
    Write a list of SMILES to a line-delimited file.
    """
    # write sampled SMILES
    with open(smiles_file, 'w') as f:
        for sm in smiles:
            _ = f.write(sm + '\n')

def read_canonical_smiles(smiles_file):
    """
    Read a list of canonical SMILES in directly from a line-delimited file,
    in a manner that avoids storing all RDKit molecule objects in memory.
    """
    original_smiles = read_smiles(smiles_file)
    canonical_smiles = []
    for smiles in original_smiles:
        mol = clean_mol(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            canonical_smiles.append(canonical)

    return canonical_smiles

def decrease_learning_rate(optimizer, multiplier=0.99):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= multiplier

def print_update(model, dataset, epoch, batch_idx, training_loss, batch_size,
                 selfies=False):
    """
    Print an update on model training, including the current epoch,
    current step (minibatch), training loss, validation loss, a selection of
    sampled SMILES, and the proportion of those SMILES that are valid.

    Args:
        model: the model currently being trained
        dataset: the dataset being trained on
        epoch: current epoch of model training
        batch_idx: current step (minibatch index) within the current epoch, or
            'NA' at the end of an epoch
        training_loss: training loss at the last step
        batch_size: size of each minibatch; used to calculate validation loss
            and sample SMILES
    """
    # calculate validation loss
    validation, lengths = dataset.get_validation(batch_size)
    validation_loss = model.loss(validation, lengths).mean().detach().item()

    # print message
    tqdm.write("*" * 50)
    tqdm.write("epoch {:3d} -- step {:3d} -- loss: {:5.2f} -- ".\
               format(epoch, batch_idx, training_loss) + \
               "validation loss: {:5.2f}".\
               format(validation_loss))

    # sample a batch of SMILES and print them
    n_smiles = batch_size
    smiles = model.sample(n_smiles, return_smiles=True)
    valid = 0
    for idx, sm in enumerate(smiles):
        if idx < 5:
            tqdm.write(sm)
        if selfies:
            sf = sm
            sm = decoder(sf)
        if sm is not None and Chem.MolFromSmiles(sm):
            valid += 1
        else:
            ## print 'invalid' SMILES
            if selfies:
                tqdm.write("invalid SELFIES: {}".format(sf))
    pct_valid = 100 * valid / n_smiles
    tqdm.write("{:>4.1f}% valid SMILES".format(pct_valid) + "\n")

def track_loss(output_file, model, dataset, epoch, step_idx,
               training_loss, batch_size):
    """
    Log model training and validation losses to a file.

    Args:
        output_file: the file to write the training schedule to
        model: the model currently being trained
        dataset: the dataset being trained on
        epoch: current epoch of model training
        step_idx: current step (minibatch index) overall
        training_loss: training loss at the last step
        batch_size: size of each minibatch; used to calculate validation loss
            and sample SMILES
    """

    sched = pd.DataFrame()
    validation, lengths = dataset.get_validation(batch_size)
    validation_loss = model.loss(validation, lengths).mean().detach().item()
    sched = pd.DataFrame({'epoch': epoch + 1, 'step': step_idx,
                          'outcome': ['training loss', 'validation loss'],
                          'value': [training_loss, validation_loss]})

    # write training schedule (write header if file does not exist)
    if not os.path.isfile(output_file) or step_idx == 0:
        sched.to_csv(output_file, index=False)
    else:
        sched.to_csv(output_file, index=False, mode='a', header=False)

def sample_smiles(output_dir, sample_idx, model, sample_size, epoch,
                  step_idx):
    """
    Sample a set of SMILES from a trained model, and write them to a file

    Args:
        output_dir: directory to write output files to
        sample_idx: index of the SMILES sample being trained on; included in
            all output fles
        model: the model currently being trained
        sample_size: the number of SMILES to sample and write
        epoch: current epoch of model training
        step_idx: current step (minibatch index) overall, or 'NA' at the end
            of an epoch
    """

    sampled_smiles = []
    while len(sampled_smiles) < sample_size:
        sampled_smiles.extend(model.sample(100, return_smiles=True))

    # set up output filename
    if step_idx == "NA":
        # writing by epoch: don't include batch index
        smiles_filename = "sample-" + str(sample_idx + 1) + \
            "-epoch=" + str(epoch + 1) + "-SMILES.smi"
    else:
        # writing by step: calculate overall step
        smiles_filename = "sample-" + str(sample_idx + 1) + \
            "-epoch=" + str(epoch + 1) + "-step=" + str(step_idx) + \
            "-SMILES.smi"

    # write to file
    smiles_file = os.path.join(output_dir, smiles_filename)
    write_smiles(sampled_smiles, smiles_file)

"""
rdkit contributed code to neutralize charged molecules;
obtained from:
    https://www.rdkit.org/docs/Cookbook.html
    http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg02669.html
"""
def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

_reactions=None
def NeutraliseCharges(mol, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol
