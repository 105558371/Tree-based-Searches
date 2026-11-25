from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df


def test_recorded_table_charts_and_missing_evidence(ui):
    app, calls = ui
    app.radio(key="page").set_value("Model Evaluation").run()
    next(w for w in app.selectbox if w.label == "Metric").set_value("RMSE").run()
    next(w for w in app.selectbox if w.label.startswith("Evaluation horizon")).set_value(45).run()
    app.multiselect[0].set_value(["gru", "gwn"]).run()
    table = app.dataframe[0].value
    assert list(table.Model) == ["GRU", "Graph WaveNet"] and list(table.RMSE) == [47, 47]
    chart = app.get("vega_lite_chart")[0].proto
    source = convert_arrow_bytes_to_pandas_df(chart.datasets[0].data.data)
    assert source.RMSE.tolist() == [47, 47]
    next(w for w in app.selectbox if w.label == "Inspect model").set_value("gwn").run()
    assert any("history N/A" in c.value for c in app.caption) and app.button[0].disabled and not calls
    next(w for w in app.selectbox if w.label == "Inspect model").set_value("gru").run()
    app.button[0].click().run()
    assert len(calls) == 1 and app.dataframe[-1].value.Observed.isna().sum() == 1
    app.selectbox(key="dataset").set_value("assignment_2006").run()
    assert any("No verified evaluation" in m.value for m in app.info)
    assert not app.get("vega_lite_chart") and not app.exception
