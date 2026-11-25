import pytest


@pytest.mark.integration
def test_real_gru(smoke):
    smoke("gru", "GRU")
