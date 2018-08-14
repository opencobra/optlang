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
from optlang.interface.constraint import Constraint

EXPRESSIONS = [
    Integer(5),
    Real(3.3),
    2 * Variable("x"),
    Variable("x") + 3 * Variable("y")
]
CONTINUOUS_BOUNDS = [-1023, -33.3, None, 33.3, 1023]


class TestConstraint(object):
    """Thoroughly test the constraint class."""

    @pytest.mark.parametrize("expr, name, lb, ub", [
        (Integer(5), "foo", -1023, 33.3),
        (Real(3.3), "bar", 77.7, None),
        (2 * Variable("x"), "baz", None, -1000),
        (Variable("x") + 3 * Variable("y"), "foobar", None, None)
    ])
    def test_clone(self, expr, name, lb, ub):
        constr = Constraint(expr, name=name, lb=lb, ub=ub)
        new = Constraint.clone(constr)
        assert new is not constr
        assert new.name == constr.name
        assert new.lb == constr.lb
        assert new.ub == constr.ub

    @pytest.mark.skip("Not implemented yet in v 2.0.")
    def test_indicator_variable(self):
        pass

    @pytest.mark.skip("Not implemented yet in v 2.0.")
    def test_active_when(self):
        pass

    @pytest.mark.parametrize("kwargs, expected", [
        pytest.mark.raises(
            ({"expression": Variable("x") + 3, "name": "foo"}, None),
            exception=ValueError, message="canonical form"),
        ({"expression": Variable("x") + 3, "lb": -10, "name": "foobar"},
         "foobar: -13.0 <= x"),
        ({"expression": Variable("x") + 3, "ub": 10, "name": "bar"},
         "bar: x <= 7.0"),
        ({"expression": Variable("x") + 2, "lb": -5, "ub": 5, "name": "baz"},
         "baz: -7.0 <= x <= 3.0"),
    ])
    def test_dunder_str(self, kwargs, expected):
        constr = Constraint(**kwargs)
        assert str(constr) == expected

    @pytest.mark.skip("Not implemented yet in v 2.0.")
    def test_to_dict(self):
        pass

    @pytest.mark.skip("Not implemented yet in v 2.0.")
    def test_from_dict(self):
        pass
