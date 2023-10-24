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


