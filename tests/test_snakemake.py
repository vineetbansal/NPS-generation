import os
import os.path
from pathlib import Path
import snakemake
import tempfile
import hashlib


base_dir = Path(__file__).parent.parent

snakefile = base_dir / "snakemake/Snakefile"
config_file = base_dir / "snakemake/config_fast.json"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"


def test_snakemake():
    with tempfile.TemporaryDirectory() as temp_dir:
        success = snakemake.snakemake(
            snakefile=snakefile,
            cores=1,
            configfiles=[config_file],
            config={"random_seed": 5831, "dataset": dataset, "pubchem_tsv_file": pubchem_tsv_file, "output_dir": temp_dir},
            dryrun=False,
            latency_wait=60,
            forceall=True,
            workdir=os.path.dirname(snakefile),  # TODO: Only needed till rules call scripts, not nps commands
            verbose=True
        )
        assert success, "Snakemake did not complete successfully"

        output_dir = os.path.join(os.path.join(snakefile), temp_dir)
        ranks_file_overall = f"{output_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_ranks_structure.csv"
        checksum = hashlib.md5(''.join(open(ranks_file_overall, "r").readlines()).encode('utf8')).hexdigest()
        assert checksum == "eb1b8299b54fc36eef4f067ac0819d7d"

        tc_file_overall = f"{output_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv"
        checksum = hashlib.md5(''.join(open(tc_file_overall, "r").readlines()).encode('utf8')).hexdigest()
        assert checksum == "c6e24fa270b159239835f83ace71ff1f"
