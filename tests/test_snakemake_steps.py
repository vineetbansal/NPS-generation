from pathlib import Path
import tempfile
import hashlib

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

base_dir = Path(__file__).parent.parent

test_dir = base_dir / "tests/test_data/snakemake_output"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"


def assert_checksum(file, checksum):
    assert (
        hashlib.md5("".join(open(file, "r").readlines()).encode("utf8")).hexdigest()
        == checksum
    )


def test_00_preprocess():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        preprocess.preprocess(
            input_file=dataset,
            output_file=temp_dir / "preprocessed.smi",
            max_input_smiles=1000,
        )
        assert_checksum(
            temp_dir / "preprocessed.smi", "a7911f5e21e07911b8898c83f53cad17"
        )


def test_01_create_training_sets():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        create_training_sets.create_training_sets(
            input_file=test_dir / "prior/raw/LOTUS_truncated.txt",
            train_file=temp_dir / "train_file_{fold}",
            vocab_file=temp_dir / "vocabulary_file_{fold}",
            test_file=temp_dir / "test_file_{fold}",
            enum_factor=0,
            folds=3,
            representation="SMILES",
            min_tc=0,
            seed=5831,
            max_input_smiles=1000,
        )
        assert_checksum(temp_dir / "train_file_0", "5fb4ff667b8d4e3c7485c4755f775104")
        assert_checksum(
            temp_dir / "vocabulary_file_0", "cbe3bbefa44233ccb5c5b36150f5efc6"
        )
        assert_checksum(temp_dir / "test_file_0", "078876be8b883513f56ed95e8d0008a0")


def test_02_train_models_RNN():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        inner_train_models_RNN.train_models_RNN(
            database="LOTUS_truncated",
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
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_0_loss.csv",
            "c350cb54e87fc6975bc57cefc3a8950a",
        )


def test_03_sample_molecules_RNN():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_sample_molecules_RNN.sample_molecules_RNN(
            database="LOTUS_truncated",
            representation="SMILES",
            seed=0,
            rnn_type="LSTM",
            embedding_size=32,
            hidden_size=256,
            n_layers=3,
            dropout=0,
            batch_size=64,
            learning_rate=0.001,
            sample_mols=100,
            input_file=None,
            time_file=None,
            vocab_file=test_dir
            / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.vocabulary",
            model_file=test_dir / "0/prior/models/LOTUS_truncated_SMILES_0_0_model.pt",
            output_file=temp_dir / "LOTUS_truncated_SMILES_0_0_samples.csv",
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_0_samples.csv",
            "ed8ed8614c2922d3458067eba5318a00",
        )


def test_04_tabulate_molecules():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_tabulate_molecules.tabulate_molecules(
            input_file=test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_0_0_samples.csv",
            representation="SMILES",
            train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
            output_file=temp_dir / "LOTUS_truncated_SMILES_0_0_samples_masses.csv",
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_0_samples_masses.csv",
            "b106ac6b78a4602d8ac029e02479458b",
        )


def test_05_collect_tabulated_molecules():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/samples"
        inner_collect_tabulated_molecules.collect_tabulated_molecules(
            input_files=[
                test_dir
                / "0/prior/samples/LOTUS_truncated_SMILES_0_0_samples_masses.csv",
                test_dir
                / "0/prior/samples/LOTUS_truncated_SMILES_0_1_samples_masses.csv",
            ],
            output_file=temp_dir / "LOTUS_truncated_SMILES_0_unique_masses.csv",
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_unique_masses.csv",
            "b106ac6b78a4602d8ac029e02479458b",
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
            cv_file=[
                test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_0.smi",
                test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_1.smi",
                test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_2.smi",
            ],
            output_file=temp_dir / "LOTUS_truncated_SMILES_processed_freq-avg.csv",
            summary_fn="freq_avg",
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_processed_freq-avg.csv",
            "32c00675ac2e6b08d798579eeb7a0e86",
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
            err_ppm=10,
            seed=5831,
            chunk_size=100000,
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_structure.csv",
            "23ec8c0dfe23d41d8cf054f41b811d84",
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_CV_tc.csv",
            "01c39b3dc2e9e8840ecda5c4b39efad1",
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
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_0_CV_ranks_formula.csv",
            "586b1feed685bd424d54dc812d3733f0",
        )


def test_08_write_structural_prior_CV():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "0/prior/structural_prior"
        inner_write_structural_prior_CV.write_structural_prior_CV(
            ranks_file=temp_dir
            / "LOTUS_truncated_SMILES__all_freq-avg_CV_ranks_structure.csv",
            tc_file=temp_dir / "LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv",
            train_file=test_dir / "0/prior/inputs/train_LOTUS_truncated_SMILES_all.smi",
            test_file=test_dir / "0/prior/inputs/test_LOTUS_truncated_SMILES_all.smi",
            pubchem_file=pubchem_tsv_file,
            sample_file=test_dir
            / "0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
            err_ppm=10,
            seed=5831,
            chunk_size=100000,
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES__all_freq-avg_CV_ranks_structure.csv",
            "eb1b8299b54fc36eef4f067ac0819d7d",
        )
        assert_checksum(
            temp_dir / "LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv",
            "c6e24fa270b159239835f83ace71ff1f",
        )
