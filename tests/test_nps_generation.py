import os
import hashlib
import tempfile
from NPS_generation.clean_SMILES import main as clean_SMILES_main
from NPS_generation.augment_SMILES import main as augment_SMILES_main
from NPS_generation.train_model import main as train_model_main
from NPS_generation.calculate_outcomes import main as calculate_outcomes_main
from NPS_generation.calculate_outcomes_distribution import main as calculate_outcomes_distribution_main
from NPS_generation.sample_molecules import main as sample_molecules_main
from NPS_generation.tabulate_molecules import main as tabulate_molecules_main
import NPS_generation.data as data_folder

test_dir = os.path.join(os.path.dirname(__file__), "test_data")
data_dir = data_folder.__path__[0]
original_file = os.path.join(data_dir, "chembl_28_2000.smi")


def test_clean_SMILES():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output1.smi")
        clean_SMILES_main(original_file, output_file)

        baseline_file = os.path.join(test_dir, "output_step1.smi")
        baseline_checksum = hashlib.md5(
            ''.join(sorted(open(baseline_file, "r").readlines())).encode('utf8')).hexdigest()
        output_checksum = hashlib.md5(''.join(sorted(open(output_file, "r").readlines())).encode('utf8')).hexdigest()

        assert baseline_checksum == output_checksum


def test_augment_SMILES():
    input_file = os.path.join(test_dir, "output_step1.smi")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output2.smi")
        args_list = ['--input_file', input_file, '--output_file', output_file, '--enum_factor', '1']
        augment_SMILES_main(args_list)


def test_train_model():
    input_file = os.path.join(test_dir, "output_step2.smi")

    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [
            "--smiles_file", input_file,
            "--output_dir", temp_dir,
            "--rnn_type", "RNN",
            "--hidden_size", "16",
            "--n_layers", "1",
            "--dropout", "0.0",
            "--bidirectional", "False",
            "--nonlinearity", "tanh",
            "--learning_rate", "1.0",
            "--learning_rate_decay", "0.1",
            "--seed", "23",
            "--batch_size", "16",
            "--max_epochs", "4",
            "--patience", "10",
            "--sample_idx", "42",
            "--sample_every_epochs", "200",
            "--sample_every_steps", "200",
            "--log_every_steps", "1",
            "--sample_size", "200",
        ]

        train_model_main(args_list)


def test_calculate_outcomes():
    sampled_file = os.path.join(test_dir, "input_step4.smi")

    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [
            '--original_file', original_file,
            '--output_dir', temp_dir,
            '--sampled_files', sampled_file
        ]
        calculate_outcomes_main(args_list)


def test_calculate_outcomes_distribution():
    sampled_file = os.path.join(test_dir, "input_step4.smi")

    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [
            '--smiles_file', sampled_file,
            '--reference_file', original_file,
            '--output_dir', temp_dir
        ]
        calculate_outcomes_distribution_main(args_list)


def test_sample_molecules():
    model_file = os.path.join(test_dir, "model-1.pt")
    smiles_file = os.path.join(test_dir, "output_step1.smi")

    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [
            '--model_file', model_file,
            '--smiles_file', smiles_file,
            '--output_dir', temp_dir,
            '--mols_per_file', '1000',
            '--sample_idx', '1',
            '--batch_size', '32'
        ]
        sample_molecules_main(args_list)



def test_tabulate_molecules():
    input_files = os.path.join(test_dir, "input_step7.smi")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output_step7.smi")
        args_list = [
            '--output_file', output_file,
            '--input_files', input_files,
        ]
        tabulate_molecules_main(args_list)