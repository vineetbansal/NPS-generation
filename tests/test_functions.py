import contextlib
from pathlib import Path
import pytest
import numpy as np
import seaborn as sns
import tempfile
import pandas as pd
from clm.functions import read_file, read_csv_file, write_to_csv_file


@contextlib.contextmanager
def local_seed(seed):
    current_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(current_state)


def test_read_file_all(dataset):
    data = read_file(
        dataset, max_lines=None, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 2501


def test_read_file_5(dataset):
    data = read_file(
        dataset, max_lines=5, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 5
    assert data[3] == "CS(=O)(=O)C=CCO"  # Line 4


def test_read_file_stream(dataset):
    data = read_file(
        dataset, max_lines=None, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 2501


def test_read_file_stream_5(dataset):
    data = read_file(
        dataset, max_lines=5, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 5


def test_read_file_5_randomize(dataset):
    with local_seed(1234):
        data = read_file(
            dataset, max_lines=5, smile_only=False, stream=False, randomize=True
        )
        assert len(data) == 5
        assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_stream_5_randomize(dataset):
    with local_seed(1234):
        data = read_file(
            dataset, max_lines=5, smile_only=False, stream=True, randomize=True
        )

    with pytest.raises(TypeError):
        len(data)

    data = list(data)
    assert len(data) == 5
    assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_all_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed,
        max_lines=None,
        smile_only=False,
        stream=False,
        randomize=False,
    )
    assert len(data) == 2501


def test_read_file_5_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed, max_lines=5, smile_only=False, stream=False, randomize=False
    )
    assert len(data) == 5
    assert data[3] == "CS(=O)(=O)C=CCO"  # Line 4


def test_read_file_stream_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed,
        max_lines=None,
        smile_only=False,
        stream=True,
        randomize=False,
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 2501


def test_read_file_stream_5_compressed(dataset_compressed):
    data = read_file(
        dataset_compressed, max_lines=5, smile_only=False, stream=True, randomize=False
    )
    with pytest.raises(TypeError):
        len(data)
    assert len(list(data)) == 5


def test_read_file_5_randomize_compressed(dataset_compressed):
    with local_seed(1234):
        data = read_file(
            dataset_compressed,
            max_lines=5,
            smile_only=False,
            stream=False,
            randomize=True,
        )
        assert len(data) == 5
        assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_read_file_stream_5_randomize_compressed(dataset_compressed):
    with local_seed(1234):
        data = read_file(
            dataset_compressed,
            max_lines=5,
            smile_only=False,
            stream=True,
            randomize=True,
        )

    with pytest.raises(TypeError):
        len(data)

    data = list(data)
    assert len(data) == 5
    assert data[3] == "CC1(C)C=Cc2c(cc(CO)c(C(=O)O)c2O)O1"  # Line 517


def test_write_csv_uncompressed_dataframe():
    iris = sns.load_dataset("iris")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "iris.csv", iris)
        df = read_csv_file(temp_dir / "iris.csv")
        pd.testing.assert_frame_equal(iris, df)


def test_write_csv_compressed_dataframe():
    iris = sns.load_dataset("iris")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "iris.csv.gz", iris)
        df = read_csv_file(temp_dir / "iris.csv.gz")
        pd.testing.assert_frame_equal(iris, df)


def test_write_csv_uncompressed_iterable():
    data = ["foo", "bar", "baz"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "data.csv", data, string_format="{}\n")
        df = read_csv_file(temp_dir / "data.csv", header=None)
        pd.testing.assert_frame_equal(pd.DataFrame(data), df)


def test_write_csv_compressed_iterable():
    data = ["foo", "bar", "baz"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "data.csv.gz", data, string_format="{}\n")
        df = read_csv_file(temp_dir / "data.csv.gz", header=None)
        pd.testing.assert_frame_equal(pd.DataFrame(data), df)


def test_write_csv_uncompressed_dataframe_append():
    iris = sns.load_dataset("iris")
    iris2 = iris.copy()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "iris.csv", iris)
        write_to_csv_file(temp_dir / "iris.csv", iris2, mode="a+")
        df = read_csv_file(temp_dir / "iris.csv")
        pd.testing.assert_frame_equal(
            pd.concat([iris, iris2]).reset_index(drop=True), df
        )


def test_write_csv_compressed_dataframe_append():
    iris = sns.load_dataset("iris")
    iris2 = iris.copy()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "iris.csv.gz", iris)
        write_to_csv_file(temp_dir / "iris.csv.gz", iris2, mode="a+")
        df = read_csv_file(temp_dir / "iris.csv.gz")
        pd.testing.assert_frame_equal(
            pd.concat([iris, iris2]).reset_index(drop=True), df
        )


def test_write_csv_uncompressed_iterable_append():
    data = ["foo", "bar", "baz"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "data.csv", data, string_format="{}\n")
        write_to_csv_file(temp_dir / "data.csv", data, string_format="{}\n", mode="a+")
        df = read_csv_file(temp_dir / "data.csv", header=None)
        pd.testing.assert_frame_equal(
            pd.concat([pd.DataFrame(data), pd.DataFrame(data)]).reset_index(drop=True),
            df,
        )


def test_write_csv_compressed_iterable_append():
    data = ["foo", "bar", "baz"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        write_to_csv_file(temp_dir / "data.csv.gz", data, string_format="{}\n")
        write_to_csv_file(
            temp_dir / "data.csv.gz", data, string_format="{}\n", mode="a+"
        )
        df = read_csv_file(temp_dir / "data.csv.gz", header=None)
        pd.testing.assert_frame_equal(
            pd.concat([pd.DataFrame(data), pd.DataFrame(data)]).reset_index(drop=True),
            df,
        )
