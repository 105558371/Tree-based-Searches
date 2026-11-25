import pandas as pd
import pytest
from services import data_service as ds, model_service as models


def test_schema_order_and_compatibility():
    rows = pd.DataFrame([{"Date": pd.Timestamp("2006-10-01"), "SCATS Number": site,
                          **{v: i+site for i, v in enumerate(ds.SLOTS)}} for site in (3002, 2000)])
    wide = ds.wide_counts(rows)
    assert list(wide) == [2000, 3002] and wide.shape == (96, 2)
    assert wide.at[pd.Timestamp("2006-10-01 23:45"), 2000] == 2095
    for malformed in (rows.drop(columns="V12"), pd.concat([rows, rows]), rows.assign(V06=-1)):
        with pytest.raises(ValueError):
            ds.wide_counts(malformed)
    with pytest.raises(ds.Unavailable, match="2006"):
        models.forecast(ds.ASSIGNMENT, "lstm", "2006-10-01 08:00")
    actual, days = ds.catalog(ds.ASSIGNMENT)
    assert actual.site_id.is_monotonic_increasing and 2000 in actual.site_id.values and days[0].year == 2006
