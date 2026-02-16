LOCATION_MAP = {
    "New York": 0,
    "London": 1,
    "Berlin": 2,
    "Mumbai": 3,
    "Unknown": -1,
}


def encode_location(location):
    """
    Encode location into numerical value for RL
    """
    return LOCATION_MAP.get(location, -1)


def regional_bias(location):
    """
    Optional: return region-level bias for song filtering
    """
    if location in ["Mumbai"]:
        return {"language": "Hindi"}
    if location in ["Berlin"]:
        return {"genre": "Techno"}
    return {}
