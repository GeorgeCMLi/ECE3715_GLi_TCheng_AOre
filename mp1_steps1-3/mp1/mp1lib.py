"""ECE 3715 Mini-Project 1 functions. Step 1 is implemented here."""

import numpy as np


def bsc(bits, p, rng):
    """Return bits after independent flips with probability p.

    Parameters
    ----------
    bits : array-like
        An array containing only zeros and ones; any shape is supported.
    p : float
        Per-bit error probability, from zero to one inclusive.
    rng : numpy.random.Generator
        The random generator assigned to the calling step.

    Returns
    -------
    numpy.ndarray
        Received bits as uint8, with the same shape as bits.
    """
    bits = np.asarray(bits)
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1.")

    flips = rng.random(bits.shape) < p
    return np.bitwise_xor(bits.astype(np.uint8), flips.astype(np.uint8))
