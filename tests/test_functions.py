import contextlib
from pathlib import Path
import pytest
import numpy as np
from clm.functions import read_file


base_dir = Path(__file__).parent.parent
dataset = base_dir / "tests/test_data/LOTUS_truncated.txt"


@contextlib.contextmanager
def local_seed(seed):
    current_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(current_state)


def test_read_file_all():
    data = read_file(
        dataset, max_lines=None, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 2501


def test_read_file_5():
    data = read_file(
        dataset, max_lines=5, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 5
    assert data[3] == "CS(=O)(=O)C=CCO"  # Line 4


def test_read_file_stream():
    data = read_file(
        dataset, max_lines=None, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 2501


def test_read_file_stream_5():
    data = read_file(
        dataset, max_lines=5, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 5


def test_read_file_5_randomize():
    with local_seed(1234):
        data = read_file(
            dataset, max_lines=5, smile_only=False, stream=False, randomize=True
        )
        assert len(data) == 5
        assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_stream_5_randomize():
    with local_seed(1234):
        data = read_file(
            dataset, max_lines=5, smile_only=False, stream=True, randomize=True
        )

    with pytest.raises(TypeError):
        len(data)

    data = list(data)
    assert len(data) == 5
    assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517
