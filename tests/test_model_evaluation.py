from pathlib import Path
import tempfile
import pandas as pd
import os.path

from clm.commands import calculate_outcomes

base_dir = Path(__file__).parent.parent
test_dir = base_dir / "tests/test_data"


def test_calculate_outcomes():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "calculate_outcome.csv"
        outcomes = calculate_outcomes.calculate_outcomes(
            train_file=test_dir / "prep_outcomes_freq.csv",
            sampled_dir=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
            output_file=output_file,
            max_orig_mols=10000,
            seed=12,
        )

        # ignore leading folders of the filename
        outcomes["input_file"] = outcomes["input_file"].apply(
            lambda path: os.path.basename(path)
        )

        true_outcomes = pd.read_csv(test_dir / "calculate_outcome.csv")
        pd.testing.assert_frame_equal(outcomes, true_outcomes)
