# -*- coding: utf-8 -*-

# Copyright 2017 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

from itertools import product

import pytest

from optlang.symbols import Mul, Integer, Real
from optlang.interface import Variable, SymbolicParameter
from optlang.interface.optimization_expression import OptimizationExpression

EXPRESSIONS = [
    Integer(5),
    Real(3.3),
    2 * Variable("x"),
    Variable("x") + 3 * Variable("y")
]


class TestOptimizationExpression(object):
    """Thoroughly test the optimization expression class."""

    @pytest.mark.parametrize("expr", EXPRESSIONS)
    def test_init_expression(self, expr):
        OptimizationExpression(expr)

    @pytest.mark.parametrize("name", [
        "R2D2",
        pytest.mark.raises("", exception=ValueError,
                           message="must not be empty"),
        pytest.mark.raises("foo bar", exception=ValueError,
                           message="cannot contain whitespace characters"),
    ])
    def test_init_name(self, name):
        OptimizationExpression(1, name)

    @pytest.mark.parametrize("expr", EXPRESSIONS)
    def test_get_expression(self, expr):
        oexpr = OptimizationExpression(expr)
        assert oexpr.expression == expr

    @pytest.mark.xfail(reason="Not yet implemented for symbolic paramters.",
                       strict=True)
    @pytest.mark.parametrize("expr, is_lin", [
        (1, True),
        (Mul(1, 3), True),
        (1 + Variable("x"), True),
        (1 + Variable("x") ** 1, True),
        (1 + Variable("x") ** 2, False),
        (1 + SymbolicParameter("mu") * Variable("x"), True),
        (1 + Variable("y") * Variable("x"), False),
        (1 + Variable("x") ** 3, False),
        (1 + Variable("z") * Variable("y") ** Variable("x"), False),
    ])
    def test_is_linear(self, expr, is_lin):
        oexpr = OptimizationExpression(expr)
        assert oexpr.is_linear() == is_lin

    @pytest.mark.parametrize("expr, is_quad", [
        (1, False),
        (Mul(1, 3), False),
        (1 + Variable("x"), False),
        (1 + Variable("x") ** 1, False),
        (1 + Variable("x") ** 2, True),
        (1 + SymbolicParameter("mu") * Variable("x"), False),
        (1 + Variable("y") * Variable("x"), True),
        (1 + Variable("x") ** 3, False),
        (1 + Variable("z") * Variable("y") ** Variable("x"), False),
    ])
    def test_is_quadratic(self, expr, is_quad):
        oexpr = OptimizationExpression(expr)
        assert oexpr.is_quadratic() == is_quad

    @pytest.mark.parametrize("expr, other", list(
        product(EXPRESSIONS, repeat=2)))
    def test_dunder_iadd(self, expr, other):
        oexpr = OptimizationExpression(expr)
        oexpr += other
        assert oexpr.expression == expr + other

    @pytest.mark.parametrize("expr, other", list(
        product(EXPRESSIONS, repeat=2)))
    def test_dunder_isub(self, expr, other):
        oexpr = OptimizationExpression(expr)
        oexpr -= other
        assert oexpr.expression == expr - other

    @pytest.mark.parametrize("expr, other", list(
        product(EXPRESSIONS, repeat=2)))
    def test_dunder_imul(self, expr, other):
        oexpr = OptimizationExpression(expr)
        oexpr *= other
        assert oexpr.expression == expr * other

    @pytest.mark.parametrize("expr, other", list(
        product(EXPRESSIONS, repeat=2)))
    def test_dunder_idiv(self, expr, other):
        oexpr = OptimizationExpression(expr)
        # Cannot realiably trigger `__idiv__` by using `/=` since its Python 2.
        oexpr.__idiv__(other)
        assert oexpr.expression == expr / other

    @pytest.mark.parametrize("expr, other", list(
        product(EXPRESSIONS, repeat=2)))
    def test_dunder_itruediv(self, expr, other):
        oexpr = OptimizationExpression(expr)
        oexpr.__itruediv__(other)
        assert oexpr.expression == expr / other

    def test_get_linear_coefficients(self):
        oexpr = OptimizationExpression(5)
        with pytest.raises(NotImplementedError):
            oexpr.get_linear_coefficients(None)

    def test_set_linear_coefficients(self):
        oexpr = OptimizationExpression(5)
        with pytest.raises(NotImplementedError):
            oexpr.set_linear_coefficients(None)
