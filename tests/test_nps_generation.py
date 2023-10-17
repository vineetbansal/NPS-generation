import os
from NPS_generation.clean_SMILES import main as clean_SMILES_main
# from NPS_generation.util.SmilesEnumerator import SmilesEnumerator
from NPS_generation.augment_SMILES import main as augment_SMILES_main
import NPS_generation.data as data_folder

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = data_folder.__path__[0]


def test_clean_SMILES():
    input_file = os.path.join(data_dir, "chembl_28_2000.smi")
    output_file = os.path.join(test_dir, "test_data", "output1.smi")
    clean_SMILES_main(input_file, output_file)
    assert True


def test_augment_SMILES():
    input_file = os.path.join(test_dir, "test_data", "output1.smi")
    output_file = os.path.join(test_dir, "test_data", "output2.smi")
    augment_SMILES_main(input_file, output_file, enum_factor=1)
    assert True
