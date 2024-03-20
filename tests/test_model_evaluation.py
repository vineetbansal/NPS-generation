from pathlib import Path
import tempfile
import pandas as pd
import os.path
import pytest
from clm.commands.calculate_outcomes import (
    get_dicts,
    process_outcomes,
    calculate_outcomes,
)
from clm.commands.write_nn_Tc import write_nn_Tc
from clm.commands.train_discriminator import train_discriminator
from clm.commands.write_freq_distribution import write_freq_distribution

base_dir = Path(__file__).parent.parent
test_dir = base_dir / "tests/test_data"


@pytest.mark.xfail
def test_generate_outcome_dicts():
    train_dict, gen_dict = get_dicts(
        train_file=test_dir / "prep_outcomes_freq.csv",
        sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
        max_orig_mols=10000,
        seed=12,
    )

    # Certain fields in train_dict/gen_dict need to be tolerant of Nones
    train_dict["qed"][4] = None
    gen_dict["SA"][3] = None

    process_outcomes(
        train_dict,
        gen_dict,
        "out.txt",
        test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
    )


def test_calculate_outcomes():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "calculate_outcome.csv"
        outcomes = calculate_outcomes(
            train_file=test_dir / "prep_outcomes_freq.csv",
            sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
            output_file=output_file,
            max_orig_mols=10000,
            seed=12,
        )
        # ignore leading folders of the filename
        outcomes["input_file"] = outcomes["input_file"].apply(
            lambda path: os.path.basename(path)
        )

        true_outcomes = pd.read_csv(test_dir / "calculate_outcome.csv")
        # https://stackoverflow.com/questions/14224172
        pd.testing.assert_frame_equal(
            outcomes.sort_index(axis=1), true_outcomes.sort_index(axis=1)
        )


def test_write_nn_tc():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "write_nn_tc.csv"
        outcomes = write_nn_Tc(
            query_file=test_dir / "prep_nn_tc_PubChem.csv",
            reference_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
            output_file=output_file,
        )

        true_outcomes = pd.read_csv(test_dir / "write_nn_tc.csv")
        pd.testing.assert_frame_equal(outcomes, true_outcomes)


def test_write_freq_distribution():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "write_freq_distribution.csv"
        outcomes = write_freq_distribution(
            sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
            test_file=test_dir / "test_LOTUS_SMILES_all_trunc.smi",
            output_file=output_file,
        )

        true_outcomes = pd.read_csv(test_dir / "write_freq_distribution.csv")
        pd.testing.assert_frame_equal(outcomes, true_outcomes)


def test_train_discriminator():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "train_discriminator.csv"
        outcomes = train_discriminator(
            train_file=test_dir
            / "snakemake_output/0/prior/inputs/train_LOTUS_truncated_SMILES_all.smi",
            sample_file=test_dir
            / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
            output_file=output_file,
            seed=0,
        )

        true_outcomes = pd.read_csv(test_dir / "train_discriminator.csv")
        pd.testing.assert_frame_equal(outcomes, true_outcomes)
