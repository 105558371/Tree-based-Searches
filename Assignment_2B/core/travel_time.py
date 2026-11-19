"""Flow -> speed -> travel time, per the 'Traffic Flow to Travel Time Conversion v1.0' spec."""
import math

A, B = -1.4648375, 93.75        # flow = A*s^2 + B*s
SPEED_LIMIT = 60.0              # km/h
INTERSECTION_DELAY = 30.0       # seconds per controlled intersection

def flow_to_speed(flow_per_hour):
    """Speed (km/h) from hourly flow. Uses the under-capacity (green) branch."""
    if flow_per_hour <= 0:
        return SPEED_LIMIT
    disc = max(0.0, B*B - 4*(-A)*flow_per_hour)      # B^2 - 4*(-A)*flow
    speed = (-B - math.sqrt(disc)) / (2 * A)             # larger root (under capacity)
    return min(speed, SPEED_LIMIT)                       # capped at 351 veh/h and below

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def travel_time_seconds(distance_km, flow_15min):
    """Edge cost A->B. flow_15min = predicted 15-min count at the START site A."""
    speed = flow_to_speed(flow_15min * 4)                # 15-min count -> veh/h
    return distance_km / speed * 3600 + INTERSECTION_DELAY
