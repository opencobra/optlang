# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

from optlang.interface import Constraint, Variable, Objective
from optlang.expression_parsing import parse_optimization_expression

import unittest


class ExpressionParsingTestCase(unittest.TestCase):
    def setUp(self):
        self.vars = [Variable(name) for name in ["x", "y", "z", "v", "w"]]

    def test_parse_linear_expression(self):
        x, y, z = self.vars[:3]
        offset = 3
        expr = 1 * x + 2 * y - 3 * z + offset
        target = {x: 1, y: 2, z: -3}

        offset_const, linear_terms_const, quad_terms_const = parse_optimization_expression(Constraint(expr, lb=0))
        offset_obj, linear_terms_obj, quad_terms_obj = parse_optimization_expression(Objective(expr), linear=False)

        self.assertEqual(offset_const, 0)
        self.assertEqual(offset_obj, offset)
        self.assertEqual(linear_terms_const, target)
        self.assertEqual(linear_terms_obj, target)
        self.assertEqual(quad_terms_const, {})
        self.assertEqual(quad_terms_obj, {})

    def test_parse_quadratic_expression(self):
        x, y, z = self.vars[:3]

        offset = 4
        expr = 2 * x**2 + 3 * x * y - 4 * z**2 + offset
        target = {frozenset([x]): 2, frozenset([z]): -4, frozenset([x, y]): 3}

        offset_const, linear_terms_const, quad_terms_const = parse_optimization_expression(Constraint(expr, lb=0), linear=False)
        offset_obj, linear_terms_obj, quad_terms_obj = parse_optimization_expression(Objective(expr), quadratic=True)

        self.assertEqual(offset_const, 0)
        self.assertEqual(offset_obj, offset)
        self.assertEqual(linear_terms_const, {})
        self.assertEqual(linear_terms_obj, {})
        self.assertEqual(quad_terms_const, target)
        self.assertEqual(quad_terms_obj, target)

    def test_parse_non_expanded_quadratic_expression(self):
        x, y, z = self.vars[:3]

        offset = 5
        expr = (x + y)**2 - (z - 2)**2 + offset
        target = {frozenset([x]): 1, frozenset([y]): 1, frozenset([x, y]): 2, frozenset([z]): -1}
        linear_target = {z: 4}

        offset_const, linear_terms_const, quad_terms_const = parse_optimization_expression(Constraint(expr, lb=0), quadratic=True)
        offset_obj, linear_terms_obj, quad_terms_obj = parse_optimization_expression(Objective(expr), linear=False)

        self.assertEqual(offset_const, -4)
        self.assertEqual(offset_obj, -4 + offset)
        self.assertEqual(linear_terms_const, linear_target)
        self.assertEqual(linear_terms_obj, linear_target)
        self.assertEqual(quad_terms_const, target)
        self.assertEqual(quad_terms_obj, target)
