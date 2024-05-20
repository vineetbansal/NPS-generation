from pathlib import Path
import pandas as pd
import pytest
from clm.commands.calculate_outcomes import (
    get_dataframes,
    calculate_outcomes_dataframe,
    calculate_outcomes,
)
from clm.commands.inner_prep_nn_tc_PubChem import prep_nn_tc
from clm.commands.write_nn_Tc import write_nn_Tc
from clm.commands.train_discriminator import train_discriminator
from clm.commands.write_freq_distribution import write_freq_distribution
from clm.commands.calculate_outcome_distrs import calculate_outcome_distr
from clm.commands.add_carbon import add_carbon
from clm.commands.plot import plot
from clm.functions import assert_checksum_equals, read_csv_file

base_dir = Path(__file__).parent.parent
test_dir = base_dir / "tests/test_data"


@pytest.mark.xfail(strict=True)
def test_generate_outcome_dicts():
    train_df, sample_df = get_dataframes(
        train_file=test_dir / "prep_outcomes_freq.csv",
        sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
    )

    # Certain fields in train_dict/gen_dict need to be tolerant of Nones
    train_df["qed"].iloc[4] = None
    sample_df["SA"].iloc[3] = None

    calculate_outcomes_dataframe(
        train_df,
        sample_df,
    )


def test_calculate_outcomes(tmp_path):
    output_file = tmp_path / "calculate_outcomes.csv"
    outcomes = calculate_outcomes(
        sampled_file=test_dir / "prep_outcomes_freq.csv",
        # For LOTUS, train/test "_all.smi" files are the same
        train_file=test_dir / "test_LOTUS_SMILES_all_trunc.smi",
        output_file=output_file,
        seed=12,
    )

    true_outcomes = read_csv_file(
        test_dir / "calculate_outcome.csv", keep_default_na=False
    )
    # https://stackoverflow.com/questions/14224172
    pd.testing.assert_frame_equal(
        outcomes.sort_index(axis=1)
        .sort_values(["outcome", "bin"])
        .reset_index(drop=True),
        true_outcomes.sort_index(axis=1)
        .sort_values(["outcome", "bin"])
        .reset_index(drop=True),
    )

    plot(
        evaluation_type="calculate_outcomes",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_prep_nn_tc(tmp_path):
    output_file = tmp_path / "prep_nn_tc_PubChem.csv"
    outcomes = prep_nn_tc(
        sample_file=test_dir / "prep_nn_tc_input.csv",
        max_molecules=100,
        pubchem_file=test_dir / "PubChem_truncated.tsv",
        output_file=output_file,
        seed=0,
    )

    true_outcomes = read_csv_file(test_dir / "prep_nn_tc_output.csv")
    pd.testing.assert_frame_equal(
        outcomes.sort_index(axis=1)
        .sort_values(["smiles", "formula"])
        .reset_index(drop=True),
        true_outcomes.sort_index(axis=1)
        .sort_values(["smiles", "formula"])
        .reset_index(drop=True),
    )


def test_write_nn_tc(tmp_path):
    query_file = test_dir / "input_write_nn_tc_query_file.csv"
    reference_file = test_dir / "input_write_nn_tc_reference_file.csv"
    output_file = tmp_path / "output_write_nn_tc.csv"
    write_nn_Tc(
        query_file=query_file,
        reference_file=reference_file,
        output_file=output_file,
    )
    assert_checksum_equals(output_file, test_dir / "output_write_nn_tc.csv")

    plot(
        evaluation_type="write_nn_tc", outcome_files=[output_file], output_dir=tmp_path
    )


def test_write_freq_distribution(tmp_path):
    output_file = tmp_path / "write_freq_distribution.csv"
    outcomes = write_freq_distribution(
        sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
        test_file=test_dir / "test_LOTUS_SMILES_all_trunc.smi",
        output_file=output_file,
    )

    true_outcomes = read_csv_file(test_dir / "write_freq_distribution.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)

    plot(
        evaluation_type="freq_distribution",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_train_discriminator(tmp_path):
    output_file = tmp_path / "train_discriminator.csv"
    outcomes = train_discriminator(
        train_file=test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi",
        sample_file=test_dir
        / "snakemake_output/0/prior/samples/LOTUS_truncated_SMILES_processed_freq-avg.csv",
        max_mols=50000,
        output_file=output_file,
        seed=0,
    )

    true_outcomes = read_csv_file(test_dir / "train_discriminator.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)

    plot(
        evaluation_type="train_discriminator",
        outcome_files=[output_file],
        output_dir=tmp_path,
    )


def test_outcome_distr(tmp_path):
    output_file = tmp_path / "outcome_distr.csv"
    outcomes = calculate_outcome_distr(
        input_file=test_dir / "write_outcome_distr.csv",
        output_file=output_file,
        seed=0,
    )

    true_outcomes = read_csv_file(test_dir / "outcome_distr.csv")
    pd.testing.assert_frame_equal(outcomes, true_outcomes)


def test_add_carbon(tmp_path):
    add_carbon(
        input_file=test_dir
        / "snakemake_output/0/prior/inputs/train0_LOTUS_truncated_SMILES_0.smi",
        output_file=tmp_path / "add_carbon.csv",
        seed=0,
    )

    assert_checksum_equals(tmp_path / "add_carbon.csv", test_dir / "add_carbon.csv")
    assert_checksum_equals(
        tmp_path / "add_carbon-unique.smi", test_dir / "add_carbon-unique.smi"
    )
