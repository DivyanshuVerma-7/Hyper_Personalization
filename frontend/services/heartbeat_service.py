def normalize_heartbeat(bpm):
    """
    Normalize BPM to [0, 1] for RL input
    """
    return min(max((bpm - 50) / (180 - 50), 0.0), 1.0)


def heartbeat_to_energy(bpm):
    """
    Map heartbeat to music energy class
    """
    if bpm < 70:
        return "low"
    elif bpm < 110:
        return "medium"
    else:
        return "high"


def preferred_tempo_range(bpm):
    """
    Returns preferred tempo range for recommendation filtering
    """
    if bpm < 70:
        return (60, 80)
    elif bpm < 110:
        return (80, 120)
    else:
        return (120, 160)
