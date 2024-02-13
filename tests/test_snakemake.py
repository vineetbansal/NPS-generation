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

        ranks_file_overall = f"{temp_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_ranks_structure.csv"
        checksum = hashlib.md5(''.join(sorted(open(ranks_file_overall, "r").readlines())).encode('utf8')).hexdigest()
        assert checksum == 'eaea0450e4f94cd7862d231164dde179'

        tc_file_overall = f"{temp_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv"
        checksum = hashlib.md5(''.join(sorted(open(tc_file_overall, "r").readlines())).encode('utf8')).hexdigest()
        assert checksum == '8d02849171f5b26ed3e90ff3cdc50d6e'