from pathlib import Path
import gzip
import pytest

base_dir = Path(__file__).parent.parent
dataset_path = base_dir / "tests/test_data/LOTUS_truncated.txt"


@pytest.fixture(scope="session")
def dataset():
    return dataset_path


@pytest.fixture(scope="session")
def dataset_compressed():
    path = dataset_path.with_suffix(".txt.gz")
    with gzip.open(path, "wb") as g:
        # Add a header line so we can exercise more code paths
        g.write("smiles\n".encode())
        with open(dataset_path, "rb") as f:
            g.write(f.read())
    yield path
