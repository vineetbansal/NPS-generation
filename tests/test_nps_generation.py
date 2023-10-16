import os
from NPS_generation.clean_SMILES import main
import NPS_generation.data as data_folder


def test_clean_SMILES():
    main_dir = data_folder.__path__
    input_file = os.path.join(main_dir[0], "chembl_28_2000.smi")
    main(input_file, output_file="output1.smi")
    assert True