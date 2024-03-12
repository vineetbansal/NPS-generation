from pathlib import Path
import tempfile
import hashlib

from clm.commands import calculate_outcomes

base_dir = Path(__file__).parent.parent
test_dir = base_dir / "tests/test_data"


def test_00_preprocess():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "calculate_outcome.csv"
        calculate_outcomes.calculate_outcomes(
            train_file=test_dir / "prep_outcomes_freq.csv",
            sampled_file=test_dir / "LOTUS_SMILES_processed_freq-avg_trunc.csv",
            output_file=output_file,
            max_orig_mols=10000,
            seed=12,
        )

        assert (
            hashlib.md5(
                "".join(open(output_file, "r").readlines()).encode("utf8")
            ).hexdigest()
            == "cab305c039f458fdc79d22804c5ba08c"
        )
