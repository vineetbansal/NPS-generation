import coverage
import os
from pathlib import Path
import snakemake
import tempfile


base_dir = Path(__file__).parent.parent

python_dir = os.path.join(base_dir, "python")
nps_generation = os.path.join(base_dir, "NPS_generation")
snakefile = os.path.join(base_dir, "snakemake/Snakefile")
config_file = os.path.join(base_dir, "snakemake/config_fast.json")
dataset = os.path.join(os.path.dirname(__file__), "test_data/LOTUS_truncated.txt")
pubchem_tsv_file = os.path.join(os.path.dirname(__file__), "test_data/PubChem_truncated.tsv")


def test_snakemake():
    with tempfile.TemporaryDirectory() as temp_dir:
        cov = coverage.Coverage(source=[python_dir, nps_generation])
        cov.start()
        result = snakemake.snakemake(
            snakefile=snakefile,
            cores=1,
            configfiles=[config_file],
            config={"dataset": dataset, "pubchem_tsv_file": pubchem_tsv_file},
            # until=["preprocess", ],
            # workdir=temp_dir,
            dryrun=False,
            latency_wait=60,
        )
        assert result, "Snakemake did not complete successfully"

        cov.stop()
        cov.save()
        percent_covered = cov.report(show_missing=True)

        # assert percent_covered == 100.0


