"""ECE 3715 Mini-Project 1 library (Team 1).

The four functions required by the project, imported by mp1.ipynb. Each has a
docstring and a hand-checkable test near the top of the notebook.

    bsc                   Steps 1-3   binary symmetric channel
    packet_errors         Steps 4, 7  bit errors in each packet
    tv_distance           Step 5      total variation distance
    attempts_until_clean  Steps 6-8   attempts until the first clean packet
"""

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


def packet_errors(n_pkt, n_bits, p, rng):
    """Number of bit errors in each of n_pkt packets of n_bits bits.

    Every bit is flipped independently with probability p, so the error count
    of a packet is Binomial(n_bits, p). The count is drawn directly from that
    distribution rather than by flipping n_pkt * n_bits individual bits: Step 7
    calls this 200 times with n_pkt = 10**5, and a bit-level version would need
    2 * 10**10 bit draws there.

    Parameters
    ----------
    n_pkt : int
        Number of packets.
    n_bits : int
        Bits per packet.
    p : float
        Bit error probability, 0 <= p <= 1.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of int, shape (n_pkt,), values in 0..n_bits.

    Hand check: p = 0 gives all zeros, p = 1 gives n_bits for every packet.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return rng.binomial(n_bits, p, size=n_pkt)


def tv_distance(pmf1, pmf2):
    """Total variation distance between two pmfs on the same support.

    Half the sum of the absolute differences:

        d_TV = 0.5 * sum_k |pmf1[k] - pmf2[k]|.

    The factor 0.5 makes the distance run from 0 (identical pmfs) to 1 (pmfs
    with disjoint support). Le Cam's bound in Step 5 uses this convention.

    Parameters
    ----------
    pmf1, pmf2 : array_like
        Probabilities over the same support, in the same order (same length).

    Returns
    -------
    float

    Hand check: identical pmfs give exactly 0; pmfs with disjoint support give
    exactly 1.
    """
    pmf1 = np.asarray(pmf1, dtype=float)
    pmf2 = np.asarray(pmf2, dtype=float)
    if pmf1.shape != pmf2.shape:
        raise ValueError("both pmfs must be over the same support (same shape)")
    return 0.5 * np.sum(np.abs(pmf1 - pmf2))


def attempts_until_clean(n_trials, q, rng):
    """Number of transmission attempts up to and including the first clean packet.

    Attempts are independent and each one is clean with probability q, so the
    count T is geometric on {1, 2, 3, ...}:

        P(T = k) = (1 - q)**(k - 1) * q,   E[T] = 1/q,   Var(T) = (1 - q)/q**2.

    Convention: this counts ATTEMPTS, so the support starts at 1. It is not the
    "failures before the first success" convention, whose support starts at 0.
    numpy's Generator.geometric uses the attempts convention, so no shift is
    applied.

    The caller supplies the per-attempt clean probability q, computed
    analytically (for an n-bit packet, q = (1 - p)**n). No bits are simulated,
    so the cost is O(n_trials) no matter how small q is.

    Parameters
    ----------
    n_trials : int
        Number of independent delivered packets (independent attempt sequences).
    q : float
        Per-attempt clean probability, 0 < q <= 1.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of int, shape (n_trials,), every value >= 1.

    Hand check: q = 1 (a perfect channel) gives exactly 1 attempt every time.
    """
    q = float(q)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    return rng.geometric(q, size=n_trials)
