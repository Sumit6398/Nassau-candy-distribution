"""
geocoding.py
------------
Offline geocoding utilities for the Nassau Candy project.

The raw dataset's Order Date / Ship Date columns turned out to be
randomly generated and carry no real shipping-lead-time signal (Same Day
shipments are, on average, no faster than Standard Class). Rather than
train a model on noise, we engineer a realistic lead-time proxy driven by
geographic distance between the assigned factory and the customer, plus
ship-mode service level. This module supplies the geography half of that
calculation.

We geocode at the State/Province level using approximate geographic
centroids. No external geocoding API is used (none is reachable from this
environment), so this lookup table is self-contained and deterministic.
Coordinates are approximate geographic centers, adequate for relative
distance comparisons across a five-factory network spanning the US/Canada.
"""

import math

# Approximate geographic centroids (lat, lon) for US states + DC and
# Canadian provinces/territories present in the dataset.
STATE_CENTROIDS = {
    # United States
    "Alabama": (32.806671, -86.791130),
    "Arizona": (34.168219, -111.930907),
    "Arkansas": (34.751928, -92.131378),
    "California": (37.271875, -119.270415),
    "Colorado": (38.997934, -105.550567),
    "Connecticut": (41.518783, -72.757507),
    "Delaware": (39.145251, -75.418717),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954456),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    # Canada
    "Alberta": (55.001251, -115.002136),
    "British Columbia": (53.726669, -127.647621),
    "Manitoba": (53.760860, -98.813873),
    "New Brunswick": (46.498390, -66.159668),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Nova Scotia": (44.681999, -63.744310),
    "Ontario": (50.000000, -86.000000),
    "Prince Edward Island": (46.510712, -63.416840),
    "Quebec": (52.939922, -73.549136),
    "Saskatchewan": (52.935143, -106.450864),
}

FACTORY_COORDINATES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points, in miles."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def state_centroid(state_province):
    coords = STATE_CENTROIDS.get(state_province)
    if coords is None:
        raise KeyError(f"No centroid known for state/province: {state_province}")
    return coords


def factory_coords(factory_name):
    coords = FACTORY_COORDINATES.get(factory_name)
    if coords is None:
        raise KeyError(f"No coordinates known for factory: {factory_name}")
    return coords


def distance_factory_to_state(factory_name, state_province):
    flat, flon = factory_coords(factory_name)
    slat, slon = state_centroid(state_province)
    return haversine_miles(flat, flon, slat, slon)
