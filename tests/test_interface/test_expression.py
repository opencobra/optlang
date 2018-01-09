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

import weakref
from itertools import product, repeat, permutations

import pytest

from optlang.symbols import UniqueSymbol
from optlang.interface.expression import OptimizationExpression


class TestOptimizationExpression(object):
    """Thoroughly test the optimization expression class."""

    @pytest.mark.parametrize("expr", [
        1,
        2 * UniqueSymbol("x")
    ])
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

    # def test_get_bounds(self, kind, lb, ub):
    #     if lb is not None and ub is not None and lb > ub:
    #         with pytest.raises(ValueError):
    #             Variable("x", type=kind, lb=lb, ub=ub)
    #     else:
    #         var = Variable("x", type=kind, lb=lb, ub=ub)
    #         assert var.lb == lb
    #         assert var.ub == ub
    #         assert var.bounds == (lb, ub)

    # def test_set_bounds(self, kind, lb, ub):
    #     var = Variable("x", type=kind)
    #     if lb is not None and ub is not None and lb > ub:
    #         with pytest.raises(ValueError):
    #             var.bounds = lb, ub
    #     else:
    #         var.bounds = lb, ub
    #         assert var.lb == lb
    #         assert var.ub == ub
    #         assert var.bounds == (lb, ub)

    # def test_set_lower_bound(self, kind, bound):
    #     var = Variable("x", type=kind, ub=0)
    #     if bound is not None and bound > var.ub:
    #         with pytest.raises(ValueError):
    #             var.lb = bound
    #     else:
    #         var.lb = bound
    #         assert var.lb == bound
    #         assert var.bounds == (bound, var.ub)

    # def test_set_upper_bound(self, kind, bound):
    #     var = Variable("x", type=kind, lb=0)
    #     if bound is not None and var.lb > bound:
    #         with pytest.raises(ValueError):
    #             var.ub = bound
    #     else:
    #         var.ub = bound
    #         assert var.ub == bound
    #         assert var.bounds == (var.lb, bound)

    # def test_primal(self, kind):
    #     var = Variable("x", type=kind)
    #     assert var.primal is None

    # def test_dual(self, kind):
    #     var = Variable("x", type=kind)
    #     assert var.primal is None

    # def test_clone(self, kind, bound):
    #     var = Variable(name="x", type=kind, lb=bound, ub=bound)
    #     new = Variable.clone(var)
    #     assert new is not var
    #     assert new.name == var.name
    #     assert new.type == var.type
    #     assert new.lb == var.lb
    #     assert new.ub == var.ub

    # def test_to_dict(self, kind, bound):
    #     var = Variable(name="x", type=kind, lb=bound, ub=bound)
    #     assert var.to_dict() == {
    #         "name": "x",
    #         "type": kind,
    #         "lb": bound,
    #         "ub": bound
    #     }

    # def test_from_dict(self, kind, bound):
    #     var = Variable.from_dict({
    #         "name": "x",
    #         "type": kind,
    #         "lb": bound,
    #         "ub": bound
    #     })
    #     assert var.name == "x"
    #     assert var.type == kind
    #     assert var.lb == bound
    #     assert var.ub == bound
