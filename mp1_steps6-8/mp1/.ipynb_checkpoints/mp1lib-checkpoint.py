"""mp1lib functions used by Steps 6-8 of ECE 3715 Mini-Project 1.

attempts_until_clean is new for Step 6: paste it into the team's mp1lib.py.

packet_errors is the Step 4 function, repeated here only so the Steps 6-8
notebook runs on its own. If mp1lib.py already has one, keep the team's version
as long as it draws the error count directly (see the docstring below); a
version that flips every bit individually will blow the 5-minute budget in Step 7.
"""
import numpy as np


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
