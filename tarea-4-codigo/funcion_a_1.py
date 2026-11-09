"""Fitness function A for Tarea 4 - Genetic Algorithm.

Contract required by the repository:
- BOUNDS defines the allowed range for every variable.
- VARIABLE_COUNT tells the genetic algorithm how many real values each member has.
- fitness returns a value to maximize.
"""

import math

BOUNDS = (-10.0, 10.0)
VARIABLE_COUNT = 1


def fitness(*variables):
    """Return sin(x)^2 / x, handling the removable limit at x = 0."""
    if len(variables) != VARIABLE_COUNT:
        raise ValueError("Function A expects exactly one variable.")

    x = float(variables[0])

    # The expression is 0/0 at x = 0, but its limit is 0.
    if abs(x) < 1e-12:
        return 0.0

    return (math.sin(x) ** 2) / x
