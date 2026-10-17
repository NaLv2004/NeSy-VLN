"""Executable low-level primitives; learning components receive only integer IDs."""
import math

# This is an actuator interface, not a predicate vocabulary or a planning rule.
PRIMITIVES = [(0.0, 0.0, 0.0)]
PRIMITIVES += [(length, yaw, 0.0) for length in (0.75, 0.35)
               for yaw in (-65.0, -40.0, -20.0, 0.0, 20.0, 40.0, 65.0)]
PRIMITIVES += [(0.20, yaw, yaw) for yaw in (-65.0, 65.0)]
N_ACTIONS = len(PRIMITIVES)


def motion(action):
    length, before, after = PRIMITIVES[int(action)]
    return {"forward_m": length, "yaw_left_rad": math.radians(before),
            "yaw_after_rad": math.radians(after)}


def validate(config):
    for length, before, after in PRIMITIVES[1:]:
        assert 0 < length <= config["agent"]["max_move_m"]
        assert max(abs(before), abs(after)) <= config["agent"]["max_turn_deg"]
