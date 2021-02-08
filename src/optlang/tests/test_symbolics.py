import unittest
import optlang


class SymbolicsTestCase(unittest.TestCase):

    def test_add_identity(self):
        self.assertEqual(optlang.symbolics.add(), 0.0)

    def test_mul_identity(self):
        self.assertEqual(optlang.symbolics.mul(), 1.0)
