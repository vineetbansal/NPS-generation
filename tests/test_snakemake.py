from pathlib import Path
import snakemake
import tempfile
import hashlib
import logging
import gzip
import clm  # noqa: F401 (imported to enable logging configuration)


base_dir = Path(__file__).parent.parent

config_file = base_dir / "snakemake/config_fast.json"
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"
pubchem_tsv_file = base_dir / "tests/test_data/PubChem_truncated.tsv"

logger = logging.getLogger("clm")


def test_snakemake():
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Running snakemake workflow in {temp_dir}")
    success = snakemake.snakemake(
        snakefile=str(base_dir / "snakemake/Snakefile"),
        cores=1,
        configfiles=[config_file],
        config={
            "random_seed": 5831,
            "paths": {
                "dataset": dataset,
                "pubchem_tsv_file": pubchem_tsv_file,
                "output_dir": temp_dir,
            },
            "representations": ["SMILES"],
            "enum_factors": [0],
            "folds": 3,
            "sample_seeds": [0],
            "structural_prior_min_freq": [1]
        },
        dryrun=False,
        latency_wait=60,
        forceall=True,
        verbose=True,
        printshellcmds=True,
    )
    assert success, "Main workflow did not complete successfully"

    ranks_file_overall = f"{temp_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_min1_all_freq-avg_CV_ranks_structure.csv.gz"
    checksum = hashlib.md5(
        "".join(
            [
                line.decode("utf8")
                for line in gzip.open(ranks_file_overall, "r").readlines()
            ]
        ).encode("utf8")
    ).hexdigest()
    assert checksum == "7d6689e0e01419d3dc7c104648b4abc9"

    tc_file_overall = f"{temp_dir}/0/prior/structural_prior/LOTUS_truncated_SMILES_min1_all_freq-avg_CV_tc.csv.gz"
    checksum = hashlib.md5(
        "".join(
            [
                line.decode("utf8")
                for line in gzip.open(tc_file_overall, "r").readlines()
            ]
        ).encode("utf8")
    ).hexdigest()
    assert checksum == "72ff5cc72f7c90c1147c0ca42748ec7e"

    # Model Evaluation Workflow
    success = snakemake.snakemake(
        snakefile=str(base_dir / "snakemake/Snakefile_model_eval"),
        cores=1,
        configfiles=[config_file],
        config={
            "paths": {
                "dataset": dataset,
                "pubchem_tsv_file": pubchem_tsv_file,
                "output_dir": temp_dir,
            },
            "representations": ["SMILES"],
            "enum_factors": [0],
            "folds": 3,
            "sample_seeds": [0],
        },
        dryrun=False,
        latency_wait=60,
        forceall=True,
        verbose=True,
        printshellcmds=True,
    )
    assert success, "Model evaluation workflow did not complete successfully"
