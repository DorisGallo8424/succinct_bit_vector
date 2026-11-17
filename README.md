# succinct_bit_vector

A compact bit vector that supports rank (count of 1s up to an index) and select (position of the n-th 1) queries, storing bits in a single Python integer.

## Usage

```python
from succinct_bit_vector import SuccinctBitVector

bv = SuccinctBitVector.from_bits([1, 0, 1, 1, 0, 0, 1])
print(len(bv))        # 7
print(bv[2])          # 1
print(bv.rank(3))     # 3  (bits 0..3 are 1,0,1,1)
print(bv.select(2))   # 3  (the third 1 is at index 3)
```

## Why this library exists

Many succinct data structure libraries maintain a rank index (superblocks) to make rank and select sub-linear. That index adds complexity and memory overhead. This library stores the bits in a single Python integer and answers queries by bit manipulation. The trade-off is that rank and select are O(n) in the worst case, but the memory footprint is minimal (one bit per stored bit plus a small constant) and the code is easy to audit. For vectors up to a few thousand bits, the simple approach is often faster because it avoids the overhead of maintaining and traversing an index.

## Edge cases

- The vector is immutable after construction.
- Negative indices are supported for `__getitem__` and `rank`, following Python sequence conventions. `select` does not accept negative indices.
- `rank(index)` is inclusive of `index`.
- `select(n)` is 0-based: `select(0)` returns the first 1.
- Bits beyond the declared length in the constructor are masked out.
