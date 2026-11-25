import pytest


@pytest.mark.integration
def test_real_lstm(smoke):
    smoke("lstm", "LSTM")
