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
    # TODO: Runs too fast - why doesn't this produce anything in temp_dir?
    with tempfile.TemporaryDirectory() as temp_dir:
        success = snakemake.snakemake(
            snakefile=snakefile,
            cores=1,
            configfiles=[config_file],
            config={"dataset": dataset, "pubchem_tsv_file": pubchem_tsv_file, "output_dir": temp_dir},
            dryrun=False,
            latency_wait=60,
            forceall=True,
            verbose=True
        )
        assert success, "Snakemake did not complete successfully"
