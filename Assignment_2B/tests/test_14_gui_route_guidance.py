import json


def test_route_widgets_map_and_summary(ui):
    app, calls = ui
    assert not app.exception and app.radio(key="page").options == ["Route Guidance", "Traffic Forecast", "Model Evaluation"]
    start = lambda: app.selectbox(key="start_site").value
    end = lambda: app.selectbox(key="end_site").value
    assert (start(), end()) == (2000, 3002) and app.select_slider[0].value == "08:00"
    app.button[0].click().run()
    assert (start(), end()) == (3002, 2000)
    app.button[1].click().run()
    assert any("No directed path" in m.value for m in app.info)
    app.button[0].click().run()
    app.segmented_control[0].set_value("Off-peak 03:00").run()
    app.slider[0].set_value(2).run()
    app.button[1].click().run()
    assert calls[-1][1].hour == 3
    assert app.metric[0].value == "4.0 min" and len(app.radio[0].options) >= 2
    routes = next(w for w in app.radio if w.label == "Route")
    assert len(routes.options) == 2
    routes.set_value(1).run()
    assert app.metric[0].value == "4.5 min" and app.metric[1].value == "3.50 km"
    deck = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
    selected = next(l for l in deck["layers"] if l.get("getColor") == [66, 133, 244])
    assert {r["route"] for r in selected["data"]} == {1}
    assert list(app.dataframe[0].value.To) == [2200, 3002]
    app.selectbox(key="end_site").set_value(2000).run()
    app.button[1].click().run()
    assert app.metric[0].value == "0.0 min"
    app.selectbox(key="dataset").set_value("assignment_2006").run()
    assert app.button[1].disabled and not app.metric
    cleared = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
    assert all(not layer["data"] for layer in cleared["layers"] if layer["@@type"] == "PathLayer")
    assert not app.exception
