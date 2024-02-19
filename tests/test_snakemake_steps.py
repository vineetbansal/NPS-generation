from pathlib import Path
import tempfile
import hashlib

from clm.commands import (
    preprocess,
    create_training_sets,
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
