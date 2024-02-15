clean-SMILES
------------

Go through each "smile" in the input file.
For each smile:
    Construct a rdkit.Chem.rdchem.Mol object
    Removes all stereochemistry info from the molecule - Chem.RemoveStereochemistry(mol)
    Sanitize molecule - Chem.SanitizeMol(mol)
    Remove any hydrogens from the graph of a molecule.

    Fragment molecule into fragments.
        Remove fragments that are <= 3 atoms (keep only heavy molecules?)

    Keep molecules that have exactly 1 fragment

    We have a list of 9 Reactants and their products
    For all these, replaces molecules matching the reactant substructure with the product

    We have 10 elements that are "valid"
    For each of the resulting molecules, keep only the ones that have only valid elements

      everything same, except
      [H]C([H])([H])Oc1cccc(-n2nc3c4cc(OC([H])([H])[H])ccc4[nH]cc-3c2=O)c1
      got replaced with
      COc1cccc(-n2nc3c4cc(OC)ccc4[nH]cc-3c2=O)c1



augment-SMILES
------------

Convert smiles into a numpy array
Create enumerator of the smiles using the SmilesEnumerator class
    - The SmilesEnumerator class: For each SMILES string in the input file, the program generates up to 'enum_factor' different SMILES strings,
        - does this by repeatedly applying the randomize_smiles
    Key Methods:
     - randomize_smiles: Performs a randomization of a SMILES string using RDKit and returns the randomized SMILES string
Generate different (randomized) SMILES strings using the 'randomize_smiles' method of SmilesEnumerator class
    - Here, different means the same molecules but different atom ordering or different representation of the same structure

    enum_factor plays a key role:
        - The program attempts to generate up to enum_factor different SMILES strings for each input SMILES string. (It might
        generate fewer than enum_factor SMILES strings if it cannot find enough unique representations within max_tries attempt)
        - The enum factor essentially determines how much the input dataset will be augmented. For example, if the enum_factor is
        set to 5, the program aims to generate 5 different SMILES strings (including the original) for each molecule in the input dataset,
        effectively augmenting the dataset size by a factor of 5

    Why do all this? By generating multiple SMILES strings for the same molecule, the program introduces diversity in the representations of the
    molecules, which can be beneficial for certain applications, such as training machine learning models on molecular data

+----------+--------------+----------------------------------+-----------------------------------+
| Case     | Enum Factor  | No. of Lines in Input File       | No. of Lines in Output File       |
|          |              |                                  | (Same output for multiples tries) |
+----------+--------------+----------------------------------+-----------------------------------+
| Case I   | 1            | 1976                             | 1976                              |
+----------+--------------+----------------------------------+-----------------------------------+
| Case II  | 5            | 1976                             | 9879                              |
+----------+--------------+----------------------------------+-----------------------------------+
| Case III | 10           | 1976                             | 19744                             |
+----------+--------------+----------------------------------+-----------------------------------+


Example:
    C=CC1(C)CCc2c3c(cc(O)c2C1O)C(C)(C)C(O)CC3
    randomizing it generates:
    First Try: C1(O)C(C)(C)c2c(c3c(c(O)c2)C(O)C(C=C)(C)CC3)CC1
    Second Try: C1C(O)C(C)(C)c2cc(O)c3c(c2C1)CCC(C=C)(C)C3O

    (All of these represent the same chemical structure, the only variation is in representation of how the atoms are ordered
    and how the rings are numbered )

 Potentially helpful stuff:
    Normal SMILES can vary for the same molecule based on the order of atoms and the path taken to traverse the molecule.
    Canonical SMILES provides a standardized and unique representation for each molecule,
    making it easier to compare and search for molecules in databases.



train_model
------------

Sets the seed for generating random number with PyTorch, the built-in Python 'random' module, and the NumPy's random number generator
Checks if CUDA is available on the system
    If true: sets the seeds for generating random numbers with PyTorch on all available GPUs
   (The steps above basically ensures that any randomness in the program behaves the same way every time its run, given the same input)
Creates a dataset (SelfiesDataset or SmilesDataset object) based on the availability of the Selfies file
Sets up a data loader that will provide batches of data from the dataset, shuffling the data, dropping the last batch if it's not full,
    and padding teh sequences in each batch so they are all the same length (based on SmilesCollate class)
Initialize a RNN model based on the provided command line argument.
    Two possible types of models:
        1. RNN with an embedding layer
        2. OneHotRNN without an embedding layer (Only runs this if embedding_size = 0)
Loads a pretrained model (if available)
Initializes an Adam optimizer which will be used to update the model's weights during training
Initializes the early stopping mechanism with the given patience value
Implements a training loop for a machine learning model, implementing several functionalities such as batch processing, loss calculation,
    gradient clipping, learning rate decay, logging, sampling, validation, and early stopping
Logs the final state of the training, recording the step at which the best model was found, and the corresponding loss value
Loads a saved model's state dictionary from a file and applies it to the model
    (typically done to restore a model to a specific state, often the one that achieved the best performance on a validation set)
Samples a set of SMILES string from the trained model until the desired sample size is reached


Classes Used:
1. SmilesDataset/ SelfiesDataset:
    Functionality:
        1. Loads the SMILES/SELFIES strings either from list or file
        2. Initializes the vocabulary, either from a provided file or by creating a new one based on the SMILES/SELFIES strings
        3. Splits the dataset into training and validation sets based on the training_split value

    Attributes:
        1. smiles: list of SMILES/SELFIES strings
        2. vocabulary: instance of the Vocabulary class used to tokenize and encode SMILES/SELFIES strings
        3. training: List of SMILES/SELFIES strings used for training (derived from the original dataset)
        4. validation: List of SMILES/SELFIES strings used for validation (derived from the original dataset)

2. SmilesCollate:
    Functionality:
        1. Sorts the batch of SMILES tensors in descending order of sequence length
        2. Pads each sequence with the previously stored padding token to make them all the same length, creating a uniform tensor
        3. Transposes the dimensions of the padded tensor to ensure that it has the shape (batch_size, seq_len)
        4. Calculates the lengths of the original sequences (before padding) and stores them in a tensor
        5. Returns the padded tensor and the tensor of sequence lengths

    Attributes:
        1. padding_token: stores the numerical value used to pad teh sequences in the batch

Definitions:
SELFIES (SELF-referencing Embedded Strings): textual representation of molecular structures designed to address some of the limitations
and challenges associate with SMILES
    - Every string generated using the SELFIES alphabet, no matter how its constructed or modified, corresponds to a valid molecular graph
     (Specially important in generative models in drug discovery, as it ensures that any modifications to a molecular structure results in a
     valid molecule)
   How is it different from SMILES?
        -> Not every string generated in the SMILES format represents a valid chemical structure. Small modifications in the string, such as
        changing a single character, can result in an invalid structure or one that does not make chemical sense

   Example:
   SMILES representation of ethane: 'CC'
        If a generative model modifies it slightly, i.e., 'C(C', it's no longer a valid SMILES. (How would a generative model know to generate a parenthesis like that?)

   SELFIES representation of ethane: [C][C]
        If a generative model modifies the string, i.e., '[C][C][]', SELFIES is designed to still produce a valid molecular graph
            - The additional bracket '[]' without a specified atom inside is simply ignored, and the molecule remains ethane


calculate_outcomes
------------

Reads the smile file generated by train_model
Constructs a list of molecule objects from the SMILES strings, replacing any invalid molecules with None.
     -uses the clean_mols function
For each molecule object in org_mols, the Chem.MolToSmiles function is called to
Convert the molecules back to a canonical SMILES string
Creates a list of lists, where each inner list contains the atomic symbols of the atoms in a molecule from the list org_mols
Calculates the unique atomic elements and their counts across all molecules in org_elements
    Example: (['Br' 'C' 'Cl' 'F' 'I' 'N' 'O' 'P' 'S'], [  103 44451   411   733     8  7016  7006    32   840])
        - The two list in the above tuple have one-to-one relation
Calculates the molecular weight of each molecule in the list org_mols and stores the results in a new list org_mws
Calculates the Bertz topological complexity (BertzCT) for each molecule in the list org_mols and stores the results in a new list org_tcs
Calculates the Topological Polar Surface Area (TPSA) for each molecule in the list org_mols and stores the results in a new list org_tpsa
Calculates the quantitative estimate of drug-likeness (QED) for a list of molecular structures and stores the results in a list
Calculates and stores the total number of rings in each molecule from the list org_mols.
Calculates and stores the total number of aliphatic (non-aromatic) rings in each molecule
        Aliphatic rings are made of carbons that are not part of an aromatic system.
Calculates and stores the total number of  aromatic rings in each molecule.
        Aromatic rings are cyclic structures with alternating single and double bonds that create a stable, resonant structure (e.g., benzene ring).
Calculates the synthetic accessibility (SA) score for a list of molecular structures and stores the results in a list
Calculates the natural product-likeness score for a list of molecules and stores the results in a list
Calculates the fraction of sp3 hybridized carbon atoms to the total number of sp2 and sp3 hybridized carbon atoms for each molecule in the list org_mols.
        In drug discovery and medicinal chemistry, a higher fraction of sp3 hybridized carbons is often associated with increased
        three-dimensional structure and complexity of the molecule, which can be a desirable property for interacting with biological targets.
Calculates the percentage of rotatable bonds for each molecule in the list org_mols, and stores the results in a new list org_rot
        This information can be important in drug design, as molecules with too many rotatable bonds may be too flexible and
        therefore may have lower binding affinity to their target, while molecules with too few rotatable bonds may be too rigid
         and might not fit well into the binding site of the target.
Calculates the Murcko scaffold for each molecule in the list org_mols and store the results in a new list org_murcko
Identifies all the unique Murcko scaffolds in the dataset and counts how many times each one appears
Calculates the number of hydrogen bond donors in each molecule in the org_mols list
        A hydrogen bond donor is an atom in a molecule that can donate a hydrogen atom to form a hydrogen bond. Typically a hydrogen atom
        attached to a highly electronegative atom like oxygen or nitrogen.
Calculates the number of hydrogen bond acceptors in each molecule in the org_mols list.
        A hydrogen bond acceptor is an atom in a molecule that can accept a hydrogen atom to form a hydrogen bond.
        Typically a highly electronegative atom like oxygen or nitrogen.
      ( The above two properties are important in drug design as they influence how a molecule interacts with biological targets,
       potentially affecting its biological activity and pharmacokinetics. The number of hydrogen bond donors and acceptors can impact
       a molecule’s solubility, permeability, and ability to bind to proteins )
Calculates the percentage of valid molecules (pct_valid) by dividing the number of valid molecules (len(gen_mols)) by the total number of generated SMILES strings (len(gen_smiles)).
Calculates the percentage of novel molecules (pct_novel) by checking how many molecules in the gen_canonical list are not in the org_canonical list.
Calculates the percentage of unique molecules (pct_unique) by finding the number of unique canonical SMILES strings in gen_canonical and dividing it by
    the total number of generated canonical SMILES strings.
Calculate various divergence measures between the generated molecules and some original set of molecules:
    K-L divergence, Jensen-Shannon distance, and Wasserstein distance of heteroatom distributions.
    K-L divergence, Jensen-Shannon distance, and Wasserstein distance of molecular weights.

....

Extracts the symbols of all atoms in each molecule of `gen_mols`, resulting in a list of lists of atom symbols.
Flattens the list of lists into a single list of atom symbols, counts the occurrences of each unique symbol, and returns two arrays:
    one for the unique atom symbols and one for their corresponding counts. This is done for the generated molecules.
Creates two dictionaries (`d1` and `d2`) mapping atom symbols to their occurrence counts for the original and generated molecules
    , respectively. It calculates two probability distributions (`p1` and `p2`), where the probability of each atom symbol is its occurrence c
    ount divided by the total count of all atom occurrences in each set.
Calculates three divergence measures between the two distributions:
   -Kullback-Leibler (KL) Divergence: Measures how one probability distribution diverges from a second, expected probability distribution. Computed using `scipy.stats.entropy`.
   -Jensen-Shannon (JS) Distance: A symmetric and smoothed version of KL divergence. Computed using `jensenshannon`.
   -Wasserstein Distance: Also known as Earth Mover’s Distance, it is a measure of the distance between two probability distributions over a region D. Computed using `wasserstein_distance`.


.....


Functions used:
1. clean_mols(all_smiles, stereochem=False, selfies=False, deepsmiles=False):
    Paramenters:
     1. all_smiles
     2. stereochem:  If False, stereochemistry information will be removed from the molecule.
     3. selfies: If True, it indicates that the input is in SELFIES format, not SMILES.
     4. deepsmiles: If True, it indicates that the input is in DeepSMILES format, not SMILES.



Definitions:
1. Stereochemistry: a branch of chemistry that involves the study of the spatial arrangement of atoms in molecules
                    and the effects of this arrangement on the properties and reactions of those molecules.
                    Basically,the 3D orientation of the atoms and the knowledge about how different atoms or
                    groups of atoms are arranged in space relative to each other.

2.DeepSMILES: an alternative representation of SMILES designed to be more compatible with deep learning models,
               with specific changes to the way rings and branches are represented to simplify the learning process

    Example: Cyclohexane
              Molecular Formula: C₆H₁₂
              Structure: A six-carbon ring with hydrogen atoms attached to each carbon.

              SMILES Representation: C1CCCCC1
              The C1...C1 denotes the ring structure, with the numeral 1 indicating the start and end of the ring.

              DeepSMILES Representation: C(CCCCC)
              Instead of using numbers to denote the start and end of the ring, DeepSMILES uses parentheses.
               The opening parenthesis ( denotes the start of the ring, and the closing parenthesis ) denotes the end.

3. Bertz topological complexity: a numerical descriptor used to quantify the structural complexity of a molecule based
            on its topological (i.e., connectivity) features.

            The idea behind this descriptor is to provide a single number that reflects the complexity of a molecule's structure,
            taking into account factors such as the number of atoms, the connectivity between atoms, and the presence of rings and branches.

           Example 1: Methane (CH₄)
               Structure: Tetrahedral, with a central carbon atom bonded to four hydrogen atoms.
               Number of Atoms: 5 (1 carbon, 4 hydrogen)
               Connectivity: Single bonds between the carbon and hydrogen atoms.
               Rings/Branches: None.
               Bertz Topological Complexity: The value would be relatively low, reflecting the simplicity of methane's structure.

           Example 2: Naphthalene (C₁₀H₈)
               Structure: Planar, consisting of two fused aromatic rings.
               Number of Atoms: 18 (10 carbon, 8 hydrogen)
               Connectivity: Alternating single and double bonds forming two fused six-membered rings.
               Rings/Branches: Two rings, with no branches.
               Bertz Topological Complexity: The value would be higher than that of methane, reflecting naphthalene's more
                                               complex aromatic structure and the presence of fused rings.


            (Can be crucial in drug discovery to filter out overly complex or overly simple compounds during virtual screening.)
                How?:  BY using Bertz complexity in virtual screening, we can aim to select compounds that strike a balance
                between being complex enough to interact specifically and strongly with the target protein,
                 but not so complex that they are synthetically inaccessible or likely to have poor bioavailability.

4. Topological Polar Surface Area: a molecular descriptor that estimates the surface area of a molecule that is accessible
            to polar solvents (like water) . It is specifically focused on polar regions of the molecule, which typically include oxygen
            and nitrogen atoms, as well as their attached hydrogen atoms.
            The TPSA is calculated based on the topological (connectivity) structure of the molecule, without the need for 3D coordinates.

            -> TPSA is particularly useful in medicinal chemistry and drug discovery for predicting the absorption, distribution,
            metabolism, excretion, and toxicity (ADMET) properties of drug candidates.


         Example 1: Nonpolar Small Molecule
         Chemical Structure: C12H26 (Dodecane)
             Functional Groups: Only carbon and hydrogen atoms; no polar functional groups.
             TPSA: 0 Å² (no polar surface area because there are no polar functional groups)
             Implications:
             Bioavailability: This compound might have good oral bioavailability due to its nonpolar nature, allowing it to pass through lipid membranes.
             Blood-Brain Barrier Penetration: The compound might be able to cross the blood-brain barrier, but its size could be a limiting factor.
             Solubility: Likely to be lipid-soluble and have poor water solubility.
             Drug-likeness: The lack of polar functional groups might limit its interaction with biological targets, reducing its effectiveness as a drug.

         Example 2: Polar Molecule with Multiple Functional Groups
         Chemical Structure: C8H9NO2 (Acetaminophen)
             Functional Groups: Hydroxyl (-OH), amide (-NH), and carbonyl (=O) groups.
             TPSA: Approximately 49.33 Å² (calculated based on the polar functional groups)
             Implications:
             Bioavailability: Good oral bioavailability, as indicated by its moderate TPSA.
             Blood-Brain Barrier Penetration: Likely to cross the blood-brain barrier due to its moderate TPSA.
             Solubility: Water-soluble due to the presence of polar functional groups, but also has lipid-soluble characteristics.
             Drug-likeness: The presence of multiple functional groups allows for diverse interactions with biological targets,
              contributing to its effectiveness as a pain reliever and fever reducer.

5. Quantitative Estimate of Drug-likeness:a metric used in medicinal chemistry to assess how "drug-like" a chemical compound is,
            based on its physicochemical properties. The concept of drug-likeness helps in identifying compounds
             that have the necessary properties to be orally active drugs in humans

6. Synthetic accessibility (SA score) : ease with which a particular chemical compound can be synthesized in the laboratory.
            It takes into account various factors including the availability of starting materials, the number of synthetic steps required,
            the yield of each step, and the complexity of the reactions.

           Compounds that are very difficult to synthesize may not be practical as drug candidates, no matter how well they perform in other tests.
           On the other hand, compounds that are easier to synthesize are more likely to be pursued further in the drug development process.

7. Natural product-likeness:  evaluation of  how closely a synthetic or virtual molecule resembles known natural products.
           Natural products have evolved to interact with biological targets, and they often have complex structures that can bind
           selectively and potently to proteins. As such, molecules that resemble natural products may also be more likely to have medicinal properties.

            Molecules with high natural product-likeness may have favorable properties in terms of absorption, distribution, metabolism, excretion, and toxicity (ADMET).



calculate_outcomes_distribution
------------


Reads SMILES & Reference SMILES strings from a file
Cleans and converts the SMILES & Reference SMILES strings to molecule objects.
Converts the  molecule objects of both SMILES & Reference SMILES back to canonical SMILES strings.
Removes any molecules that are known or are part of a reference set. The result is a list of canonical SMILES strings representing only the novel or unique molecules.
Converts the remaining SMILES strings back into RDKit molecule objects
Extracts the atomic symbols of all atoms in the molecule. The result is a list of lists, where each inner list contains the atomic symbols of a single molecule.
    - Example: [['C', 'C', 'O'], ['C', 'H', 'O']]
Counts the unique atomic symbols and their occurrences across all molecules. This function returns two arrays: one with the unique atomic symbols
    and another with the corresponding counts of each symbol.
Computes the molecular weight of each molecule in the mols list.
Computes the octanol-water partition coefficient (LogP) for each molecule.
    (LogP is a measure of a molecule's hydrophobicity, with higher values indicating more hydrophobic molecules)
Calculates the Bertz topological complexity for each molecule
Calculates the topological polar surface area (TPSA) for each molecule
Calculates the quantitative estimate of drug-likeness (QED) for a list of molecular structures and stores the results in a list
Calculates the fraction of sp3 hybridized carbons in a molecule
Calculates the percentage of heteroatoms (atoms other than carbon and hydrogen) in each molecule.
    (The number of heteroatoms is divided by the total number of atoms in the molecule to get the percentage.)
Calculates the number of rings in a molecule
Calculates the synthetic accessibility (SA) score for a list of molecular structures and stores the results in a list
Calculates the natural product-likeness score for a list of molecules and stores the results in a list
All the descriptors and properties above are added to the res DataFrame and stored in csv format



sample_molecules
-----------------

Sets the seed for generating random number with PyTorch, the built-in Python 'random' module, and the NumPy's random number generator
Checks if CUDA is available on the system
    If true: sets the seeds for generating random numbers with PyTorch on all available GPUs
Creates a dataset (SelfiesDataset or SmilesDataset object) based on the availability of the Selfies file
Initialize a RNN model based on the provided command line argument.
    Two possible types of models:
        1. RNN with an embedding layer
        2. OneHotRNN without an embedding layer (Only runs this if embedding_size = 0)
Loads a pretrained model (if available)
Generates sampled SMILES and Negative Log-Likelihood (NLL) values by calling teh 'sample' method on a 'model' object
Writes the sampled SMILES and NLL values to a file



tabulate_molecules
-------------------

Extracts the file name and extension from the user provided output file path
Creates a new temporary file name and opens it in append mode
Reads the input file line by line
Processes moleculer structures, and uses RDKit to calculate various molecular properties, such as:
    1. Mass
    2. Molecular Formula
    3. Canonical Smile
Writes the above results into a file
Reads data from the temporary file
Calculates the frequency of each unique combination of SMILE representation, mass, and formula
Extracts the best (lowest) NLL value for each unique SMILE
Combines the above information and writes the result into an output file
