import unittest

from succinct_bit_vector import SuccinctBitVector


class TestSuccinctBitVector(unittest.TestCase):
    def test_empty(self):
        bv = SuccinctBitVector()
        self.assertEqual(len(bv), 0)
        with self.assertRaises(IndexError):
            bv[0]
        with self.assertRaises(IndexError):
            bv.rank(0)
        with self.assertRaises(IndexError):
            bv.select(0)

    def test_from_bits(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1, 1, 0, 0, 1])
        self.assertEqual(len(bv), 7)
        self.assertEqual([bv[i] for i in range(7)], [1, 0, 1, 1, 0, 0, 1])

    def test_from_bits_truthiness(self):
        bv = SuccinctBitVector.from_bits([True, False, 2, 0, "x", ""])
        self.assertEqual([bv[i] for i in range(6)], [1, 0, 1, 0, 1, 0])

    def test_constructor_masks_extra_bits(self):
        # bits=0b1111, length=2 should only keep the low two bits.
        bv = SuccinctBitVector(0b1111, 2)
        self.assertEqual(len(bv), 2)
        self.assertEqual(bv[0], 1)
        self.assertEqual(bv[1], 1)
        with self.assertRaises(IndexError):
            bv[2]

    def test_negative_length_raises(self):
        with self.assertRaises(ValueError):
            SuccinctBitVector(0, -1)

    def test_getitem_negative(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1])
        self.assertEqual(bv[-1], 1)
        self.assertEqual(bv[-2], 0)
        self.assertEqual(bv[-3], 1)
        with self.assertRaises(IndexError):
            bv[-4]

    def test_rank_basic(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1, 1, 0, 0, 1])
        self.assertEqual(bv.rank(0), 1)
        self.assertEqual(bv.rank(1), 1)
        self.assertEqual(bv.rank(2), 2)
        self.assertEqual(bv.rank(3), 3)
        self.assertEqual(bv.rank(4), 3)
        self.assertEqual(bv.rank(5), 3)
        self.assertEqual(bv.rank(6), 4)

    def test_rank_negative(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1, 1, 0, 0, 1])
        self.assertEqual(bv.rank(-1), 4)
        self.assertEqual(bv.rank(-2), 3)
        self.assertEqual(bv.rank(-7), 1)
        with self.assertRaises(IndexError):
            bv.rank(-8)

    def test_rank_out_of_range(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1])
        with self.assertRaises(IndexError):
            bv.rank(3)

    def test_select_basic(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1, 1, 0, 0, 1])
        self.assertEqual(bv.select(0), 0)
        self.assertEqual(bv.select(1), 2)
        self.assertEqual(bv.select(2), 3)
        self.assertEqual(bv.select(3), 6)

    def test_select_out_of_range(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1])
        with self.assertRaises(IndexError):
            bv.select(2)
        with self.assertRaises(IndexError):
            bv.select(-1)

    def test_select_all_zeros(self):
        bv = SuccinctBitVector.from_bits([0, 0, 0])
        with self.assertRaises(IndexError):
            bv.select(0)

    def test_repr(self):
        bv = SuccinctBitVector.from_bits([1, 0, 1])
        self.assertEqual(repr(bv), "SuccinctBitVector(length=3)")

    def test_large_vector(self):
        # 1000 bits, alternating pattern.
        bits = [i % 2 for i in range(1000)]
        bv = SuccinctBitVector.from_bits(bits)
        self.assertEqual(len(bv), 1000)
        self.assertEqual(bv.rank(999), 500)
        self.assertEqual(bv.select(499), 999)
        with self.assertRaises(IndexError):
            bv.select(500)


if __name__ == "__main__":
    unittest.main()
