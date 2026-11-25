from services.routing_service import routes


def test_morning_changes_fastest_path(graph, snapshot):
    low = routes(graph, snapshot("03:00"), 2000, 3002, 1)[0]
    peak = routes(graph, snapshot("08:00"), 2000, 3002, 2)
    assert peak[0]["site_ids"] == (2000, 2200, 3002) != low["site_ids"]
    assert [r["travel_time_seconds"] for r in peak] == [180+90, 255+90]
    assert peak[0]["target_interval"][0].hour == 8
