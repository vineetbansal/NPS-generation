from pathlib import Path
import tempfile

from clm.commands import (
    preprocess,
    create_training_sets,
    inner_train_models_RNN,
    inner_sample_molecules_RNN,
    inner_tabulate_molecules,
    inner_collect_tabulated_molecules,
    inner_process_tabulated_molecules,
    inner_write_structural_prior_CV,
    inner_write_formula_prior_CV,
)
from clm.functions import assert_checksum_equals

base_dir = Path(__file__).parent.parent

test_dir = base_dir / "tests/test_data/snakemake_output"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"


def test_00_preprocess():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        preprocess.preprocess(
            input_file=dataset,
            output_file=temp_dir / "preprocessed.smi",
            max_input_smiles=1000,
        )
        assert_checksum_equals(
            temp_dir / "preprocessed.smi", test_dir / "prior/raw/LOTUS_truncated.txt"
        )


def test_01_create_training_sets():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_training_sets.create_training_sets(
            input_file=test_dir / "prior/raw/LOTUS_truncated.txt",
            train0_file=temp_dir / "train0_file_{fold}",
            train_file=temp_dir / "train_file_{fold}",
            vocab_file=temp_dir / "vocabulary_file_{fold}",
            test0_file=temp_dir / "test0_file_{fold}",
            test_file=temp_dir / "test_file_{fold}",
            enum_factor=0,
            folds=3,
            representation="SMILES",
            min_tc=0,
            seed=5831,
            max_input_smiles=1000,
        )
        # `train0_file_0` denotes the train smiles without augmentation for fold
        # 0; Since we're running with enum_factor=0, this should be identical
        # to `train_file_0` (train smiles with augmentation for fold 0)
        assert_checksum_equals(temp_dir / "train0_file_0", temp_dir / "train_file_0")
        assert_checksum_equals(
            temp_dir / "train_file_0",
            test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
        )
        assert_checksum_equals(
            temp_dir / "vocabulary_file_0",
            test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
        )
        # `test0_file_0` denotes the test smiles without augmentation for fold
        # 0; Since we're running with enum_factor=0, this should be identical
        # to `test_file_0` (test smiles with augmentation for fold 0)
        assert_checksum_equals(temp_dir / "test0_file_0", temp_dir / "test_file_0")
        assert_checksum_equals(
            temp_dir / "test_file_0",
            test_dir / "0/prior/inputs/test_LOTUS_truncated_SMILES_0.smi",
        )


def test_02_train_models_RNN():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        inner_train_models_RNN.train_models_RNN(
            representation="SMILES",
            seed=0,
            rnn_type="LSTM",
            embedding_size=32,
            hidden_size=256,
            n_layers=3,
            dropout=0,
            batch_size=64,
            learning_rate=0.001,
            max_epochs=3,
            patience=5000,
            log_every_steps=100,
            log_every_epochs=1,
            sample_mols=100,
            input_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
            vocab_file=test_dir
            / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
            model_file=temp_dir / "LOTUS_truncated_SMILES_0_0_model.pt",
            loss_file=temp_dir / "LOTUS_truncated_SMILES_0_0_loss.csv",
            smiles_file=None,
        )
        # Model loss values can vary between platforms and architectures,
        # so we simply ensure that this step runs without errors.


def test_03_sample_molecules_RNN():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_sample_molecules_RNN.sample_molecules_RNN(
            representation="SMILES",
            seed=0,
            rnn_type="LSTM",
            embedding_size=32,
            hidden_size=256,
            n_layers=3,
            dropout=0,
            batch_size=64,
            sample_mols=100,
            vocab_file=test_dir
            / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
            model_file=test_dir / "0/prior/models/LOTUS_truncated_SMILES_0_0_model.pt",
            output_file=temp_dir / "LOTUS_truncated_SMILES_0_0_0_samples.csv",
        )
        # Samples and their associated loss values can vary between platforms
        # and architectures, so we simply ensure that this step runs without
        # errors.


def test_04_tabulate_molecules():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_tabulate_molecules.tabulate_molecules(
            input_file=test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples.csv",
            representation="SMILES",
            train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
            output_file=temp_dir / "LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
            test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
        )


def test_05_collect_tabulated_molecules():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_collect_tabulated_molecules.collect_tabulated_molecules(
            input_files=[
                test_dir
                / "0/prior/samples/LOTUS_truncated_SMILES_0_0_0_samples_masses.csv",
                test_dir
                / "0/prior/samples/LOTUS_truncated_SMILES_0_1_0_samples_masses.csv",
            ],
            output_file=temp_dir / "LOTUS_truncated_SMILES_0_unique_masses.csv",
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_0_unique_masses.csv",
            test_dir / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
        )


def test_06_process_tabulated_molecules():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_process_tabulated_molecules.process_tabulated_molecules(
            input_file=[
                test_dir / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
                test_dir / "0/prior/samples/LOTUS_truncated_SMILES_1_unique_masses.csv",
                test_dir / "0/prior/samples/LOTUS_truncated_SMILES_2_unique_masses.csv",
            ],
            cv_files=[
                test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
                test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_1.smi",
                test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_2.smi",
            ],
            output_file=temp_dir / "LOTUS_truncated_SMILES_processed_freq-avg.csv",
            summary_fn="freq-avg",
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_processed_freq-avg.csv",
            test_dir / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
        )


def test_07_write_structural_prior_CV():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/structural_prior"
        inner_write_structural_prior_CV.write_structural_prior_CV(
            ranks_file=temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
            tc_file=temp_dir / "LOTUS_truncated_SMILES_0_CV_tc.csv",
            train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
            test_file=test_dir / "0/prior/inputs/test_LOTUS_truncated_SMILES_0.smi",
            pubchem_file=pubchem_tsv_file,
            sample_file=test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
            carbon_file=test_dir
            / "0/prior/inputs/train0_LOTUS_truncated_SMILES_0_carbon.csv",
            err_ppm=10,
            seed=5831,
            chunk_size=100000,
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
            test_dir
            / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_0_CV_tc.csv",
            test_dir / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_tc.csv",
        )


def test_08_write_formula_prior_CV():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        inner_write_formula_prior_CV.write_formula_prior_CV(
            ranks_file=temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
            train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
            test_file=test_dir / "0/prior/inputs/test_LOTUS_truncated_SMILES_0.smi",
            pubchem_file=pubchem_tsv_file,
            sample_file=test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_unique_masses.csv",
            err_ppm=10,
            seed=5831,
            chunk_size=100000,
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
            test_dir
            / "0/prior/structural_prior/LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
        )


def test_08_write_structural_prior_CV():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/structural_prior/add_carbon"
        inner_write_structural_prior_CV.write_structural_prior_CV(
            ranks_file=temp_dir
            / "LOTUS_truncated_SMILES_all_freq-avg_CV_ranks_structure.csv",
            tc_file=temp_dir / "LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv",
            train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_all.smi",
            test_file=test_dir / "0/prior/inputs/test_LOTUS_truncated_SMILES_all.smi",
            pubchem_file=pubchem_tsv_file,
            sample_file=test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
            carbon_file=test_dir
            / "0/prior/inputs/train0_LOTUS_truncated_SMILES_carbon_all.csv",
            err_ppm=10,
            seed=5831,
            chunk_size=100000,
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_all_freq-avg_CV_ranks_structure.csv",
            test_dir
            / "0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_ranks_structure.csv",
        )
        assert_checksum_equals(
            temp_dir / "LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv",
            test_dir
            / "0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv",
        )
