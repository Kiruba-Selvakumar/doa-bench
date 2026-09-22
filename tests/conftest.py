from pathlib import Path

import pytest

from doabench.grids import uniform_grid
from doabench.io import DOADataset, load_reference_mat

DATA = Path(__file__).resolve().parents[1] / "data" / "reference"


@pytest.fixture(scope="session")
def correlated() -> DOADataset:
    return load_reference_mat(DATA / "doa_dataset_correlated.mat")


@pytest.fixture(scope="session")
def uncorrelated() -> DOADataset:
    return load_reference_mat(DATA / "doa_dataset_uncorrelated.mat")


@pytest.fixture(scope="session")
def music_grid():
    return uniform_grid(-90, 90, 0.1)


@pytest.fixture(scope="session")
def ml_grid_1deg():
    return uniform_grid(-90, 90, 1)
