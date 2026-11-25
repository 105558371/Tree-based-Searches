import numpy as np
import pandas as pd
import pytest
from services import model_service as models


def test_replay_boundaries_and_causality(replay, snapshot):
    f = snapshot()
    assert [f.interval(h) for h in f.horizons] == [(f.origin + pd.Timedelta(minutes=h-15),
                                                     f.origin + pd.Timedelta(minutes=h)) for h in (15, 30, 45, 60)]
    for i in (12, 28):
        assert replay.position(replay.times[i], 12) == i
    for t in (replay.times[11], replay.times[29], replay.times[12]+pd.Timedelta(minutes=1)):
        with pytest.raises(ValueError):
            replay.position(t, 12)
    with pytest.raises(ValueError):
        replay.position(replay.times[12], 4*672+12)
    with pytest.raises(ValueError):
        f.snapshot(75)
    replay.mask[11, 0] = False
    before = models.inputs(replay, 12, 12)
    replay.flow[12:] += 100000
    replay.flow[11, 0] = 99999
    np.testing.assert_array_equal(before, models.inputs(replay, 12, 12))
    assert before[0, 0, 0, -1] == (8-1)/2
