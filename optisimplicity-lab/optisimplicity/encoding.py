"""Canonical encodings and practical description-length proxies."""
from __future__ import annotations

import zlib
from functools import lru_cache


def int_to_bits(value: int, n_bits: int) -> str:
    """Return a fixed-width binary representation."""
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    if value < 0 or value >= (1 << n_bits):
        raise ValueError("value does not fit in n_bits")
    return format(value, f"0{n_bits}b")


def bits_to_int(bits: str) -> int:
    if not bits or any(c not in "01" for c in bits):
        raise ValueError("bits must be a non-empty binary string")
    return int(bits, 2)


@lru_cache(maxsize=131072)
def compressed_length_bits(bits: str) -> int:
    """Length of zlib-compressed ASCII encoding, in bits.

    This is an encoding-dependent proxy, not Kolmogorov complexity.
    """
    if not bits or any(c not in "01" for c in bits):
        raise ValueError("bits must be a non-empty binary string")
    return 8 * len(zlib.compress(bits.encode("ascii"), level=9))


def all_bitstrings(n_bits: int) -> list[str]:
    if not 1 <= n_bits <= 20:
        raise ValueError("n_bits must be between 1 and 20")
    return [int_to_bits(i, n_bits) for i in range(1 << n_bits)]
