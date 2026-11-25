from services.routing_service import routes


def test_evening_recalculates_cost(graph, snapshot):
    morning, evening = [routes(graph, snapshot(t), 2000, 3002, 1)[0] for t in ("08:00", "17:00")]
    assert evening["site_ids"] == (2000, 2100, 3002) != morning["site_ids"]
    assert evening["target_interval"][0].hour == 17
    assert evening["travel_time_seconds"] == 150+142.5
