"""Conversion between the physical lengths a config states, in metres, and the element
units the mesh-level code works in.

Every element is a square of side `element_size_m` (`conventions.md`, "Mesh
assumptions"), so one length converts both axes.
"""

import math
import sys

# Relative tolerance on a length that should be a whole number of elements. Float
# rounding in a quotient of decimal lengths stays under one machine epsilon; anything a
# caller means as a fraction of an element is many orders larger.
LENGTH_RTOL = 4 * sys.float_info.epsilon


def in_elements(length_m: float, element_size_m: float) -> float:
    """
    Convert `length_m` to elements, exact where it is meant as a whole number of them:
    the continuity filter's window steps at whole numbers, so a few ulps either side of
    one would change the stencil.

    :param length_m: the length in metres
    :param element_size_m: side of a square element
    :return: the length in elements
    """
    n = length_m / element_size_m
    whole = round(n)
    return float(whole) if math.isclose(n, whole, rel_tol=LENGTH_RTOL) else n


def element_count(length_m: float, element_size_m: float, name: str) -> int:
    """The number of elements spanning `length_m`.

    :param name: what `length_m` is, for the error message
    :raises ValueError: if `length_m` is not a whole number of elements
    """
    n = in_elements(length_m, element_size_m)
    if n != round(n) or n < 1:
        raise ValueError(
            f"{name}={length_m:g} m is {n!r} square elements of side {element_size_m:g} m, not a positive whole number"
        )
    return round(n)
