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
        If a generative model modifies it slightly, i.e., 'C(C', it's no longer a valid SMILES

   SELFIES representation of ethane: [C][C]
        If a generative model modifies the string, i.e., '[C][C][]', SELFIES is designed to still produce a valid molecular graph
            - The additional bracket '[]' without a specified atom inside is simply ignored, and the molecule remains ethane