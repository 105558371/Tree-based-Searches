"""Flow to travel time, following the assignment's simplified conversion.
Flow (veh/h) and speed (km/h) are related by flow = A*v^2 + B*v, with a capacity of 1500 veh/h
reached at 32 km/h. Up to the flow of the speed limit the speed is the limit; above it we take the
under-capacity (faster) speed. Check A and B against Traffic Flow to Travel Time Conversion v1.0."""
import math

A, B, CAPACITY = -1.4648375, 93.75, 1500.0


def flow_to_speed(flow_per_hour: float, limit_kmh: float = 60.0) -> float:
    f = min(max(flow_per_hour, 0.0), CAPACITY)
    if f <= A * limit_kmh ** 2 + B * limit_kmh:
        return limit_kmh
    disc = max(B * B + 4 * A * f, 0.0)
    return min((B + math.sqrt(disc)) / (-2 * A), limit_kmh)


def travel_time_s(distance_km: float, flow_per_hour: float, delay_s: float = 30.0, limit_kmh: float = 60.0) -> float:
    return distance_km / flow_to_speed(flow_per_hour, limit_kmh) * 3600 + delay_s


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))
