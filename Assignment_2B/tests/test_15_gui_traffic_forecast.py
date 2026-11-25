import json
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df


def test_horizon_site_switch_and_unavailable(ui):
    app, calls = ui
    app.radio(key="page").set_value("Traffic Forecast").run()
    app.selectbox(key="model_research_2014_2026").set_value("gru").run()
    app.button[0].click().run()
    assert not app.exception and len(calls) == 1
    app.selectbox(key="site_research_2014_2026").set_value(2100).run()
    next(w for w in app.selectbox if w.label == "Horizon (minutes)").set_value(45).run()
    assert len(calls) == 1 and app.metric[0].value == "585.0"
    layers = json.loads(app.get("deck_gl_json_chart")[0].proto.json)["layers"]
    marker = next(r for r in layers[0]["data"] if r["site_id"] == 2100)
    assert marker["flow"] == 585 and marker["color"] == [211, 74, 64]
    assert any("Moderate" in c.value and "vehicles/15 min/site" in c.value for c in app.caption)
    assert list(app.dataframe[0].value.Predicted) == [385, 485, 585, 685]
    chart = app.get("vega_lite_chart")[0].proto
    frame = convert_arrow_bytes_to_pandas_df(chart.datasets[0].data.data)
    assert 585 in frame.select_dtypes("number").to_numpy()
    app.selectbox(key="model_research_2014_2026").set_value("gwn").run()
    assert app.button[0].disabled and not app.metric and not app.get("deck_gl_json_chart")
    assert next(w for w in app.selectbox if w.label == "Horizon (minutes)").disabled
    assert not app.exception
