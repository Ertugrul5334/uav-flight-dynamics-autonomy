import pytest

from uavsim import SixDOF, trim


@pytest.fixture(scope="session")
def model():
    return SixDOF()


@pytest.fixture(scope="session")
def level_trim(model):
    return trim(model, Va=25.0, altitude=100.0)
