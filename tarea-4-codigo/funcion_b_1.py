"""Fitness function B for Tarea 4 - Genetic Algorithm."""

import math

BOUNDS = (-10.0, 10.0)
VARIABLE_COUNT = 2


def fitness(*variables):
    """Return the value of the two-variable function that must be maximized."""
    if len(variables) != VARIABLE_COUNT:
        raise ValueError("Function B expects exactly two variables.")

    x = float(variables[0])
    y = float(variables[1])

    return (
        20.0
        + x
        - 10.0 * math.cos(2.0 * math.pi * x)
        + y
        - 10.0 * math.cos(2.0 * math.pi * y)
    )
