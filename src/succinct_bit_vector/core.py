"""A compact bit vector with rank and select operations.

This module implements a bit vector that stores bits in a Python integer
and answers rank (number of 1s up to an index) and select (index of the
n-th 1) queries.  The representation is compact: it uses one bit per
stored bit plus a small amount of metadata.  The trade-off is that rank
and select are O(n) in the worst case because we do not maintain a
superblock index.  For applications that need sub-linear queries on very
large vectors, an additional index would be required.  Here we favour
simplicity and a tiny memory footprint.
"""

from __future__ import annotations


class SuccinctBitVector:
    """A bit vector with rank and select queries.

    Bits are stored in a single Python integer, least significant bit
    first.  The bit at position ``i`` (0-based) is ``(self._bits >> i) & 1``.
    This representation is compact and avoids the overhead of a list of
    booleans or a bytearray.

    The vector is immutable after construction.  Use ``from_bits`` to
    create an instance from an iterable of 0/1 integers or booleans.
    """

    __slots__ = ("_bits", "_length")

    def __init__(self, bits: int = 0, length: int = 0) -> None:
        """Create a bit vector from an integer and a length.

        The integer ``bits`` holds the bit pattern, with the least
        significant bit corresponding to position 0.  Only the lowest
        ``length`` bits are considered; higher bits in ``bits`` are
        ignored.  This makes it easy to pass the result of parsing a
        binary string with ``int(s, 2)`` after reversing it.

        Raises:
            ValueError: If ``length`` is negative.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        self._length = length
        # Mask off any bits beyond the declared length so that queries
        # cannot accidentally see them.
        if length == 0:
            self._bits = 0
        else:
            self._bits = bits & ((1 << length) - 1)

    @classmethod
    def from_bits(cls, bits: object) -> "SuccinctBitVector":
        """Create a bit vector from an iterable of 0/1 integers or booleans.

        The iterable is consumed in order: the first element becomes bit 0,
        the second bit 1, and so on.  Any value that is truthy is stored as
        1, and any falsy value as 0.  This matches the common Python
        convention for boolean sequences.

        Args:
            bits: An iterable of values convertible to bool.

        Returns:
            A new SuccinctBitVector.
        """
        value = 0
        length = 0
        for b in bits:  # type: ignore[union-attr]
            if b:
                value |= 1 << length
            length += 1
        return cls(value, length)

    def __len__(self) -> int:
        """Return the number of bits in the vector."""
        return self._length

    def __getitem__(self, index: int) -> int:
        """Return the bit at ``index`` (0 or 1).

        Supports negative indices like a sequence.  Raises IndexError if
        the index is out of range.
        """
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("bit index out of range")
        return (self._bits >> index) & 1

    def rank(self, index: int) -> int:
        """Return the number of 1 bits in positions ``0`` through ``index``.

        The index is inclusive.  If ``index`` is negative, it is treated
        as an offset from the end of the vector, so ``rank(-1)`` counts
        all bits.  This matches Python sequence indexing and makes it
        convenient to count from the end.

        Raises:
            IndexError: If ``index`` is out of range after adjustment.
        """
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("rank index out of range")
        # Count bits in the prefix [0, index].
        mask = (1 << (index + 1)) - 1
        return (self._bits & mask).bit_count()

    def select(self, n: int) -> int:
        """Return the position of the ``n``-th 1 bit (0-based).

        ``select(0)`` returns the position of the first 1.  If there are
        fewer than ``n+1`` ones, raises IndexError.  Negative ``n`` is
        not supported and raises IndexError.

        Raises:
            IndexError: If ``n`` is negative or there is no ``n``-th 1.
        """
        if n < 0:
            raise IndexError("select index must be non-negative")
        # Walk through the bits.  This is O(length) in the worst case,
        # but the simple loop is fast in CPython for moderate sizes and
        # avoids maintaining a rank index.
        remaining = n
        bits = self._bits
        pos = 0
        while bits:
            # Isolate the lowest set bit.
            low = bits & -bits
            # Position of that bit is the number of trailing zeros.
            # bit_length() of low gives position + 1.
            bit_pos = low.bit_length() - 1
            if remaining == 0:
                return bit_pos
            remaining -= 1
            # Clear the lowest set bit.
            bits ^= low
            pos = bit_pos + 1
        raise IndexError("select index out of range")

    def __repr__(self) -> str:
        return f"SuccinctBitVector(length={self._length})"
