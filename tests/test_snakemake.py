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
            snakefile=str(snakefile),
            cores=1,
            configfiles=[config_file],
            config={
                "random_seed": 5831,
                "dataset": dataset,
                "pubchem_tsv_file": pubchem_tsv_file,
                "representations": ["SMILES"],
                "enum_factors": [0],
                "folds": 3,
                "sample_seeds": [0],
                "output_dir": temp_dir,
            },
            dryrun=False,
            latency_wait=60,
            forceall=True,
            verbose=True,
        )
        assert success, "Snakemake did not complete successfully"

        output_dir = os.path.join(os.path.join(snakefile), temp_dir)
        ranks_file_overall = f"{output_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_ranks_structure.csv"
        checksum = hashlib.md5(
            "".join(open(ranks_file_overall, "r").readlines()).encode("utf8")
        ).hexdigest()
        assert checksum == "dfa4142cae1c2f033bf31498e69189db"

        tc_file_overall = f"{output_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_all_freq-avg_CV_tc.csv"
        checksum = hashlib.md5(
            "".join(open(tc_file_overall, "r").readlines()).encode("utf8")
        ).hexdigest()
        assert checksum == "d961fa2186e16d1c2a4725a18eddda35"
