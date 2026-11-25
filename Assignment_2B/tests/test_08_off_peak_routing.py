from services.routing_service import routes


def test_low_flow_0300(graph, snapshot):
    f = snapshot("03:00")
    row = routes(graph, f, 2000, 3002, 1)[0]
    assert row["target_interval"][0].hour == 3 and row["site_ids"] == (2000, 2100, 3002)
    assert row["travel_time_seconds"] == 150 + 90
