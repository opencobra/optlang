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

import pytest

from optlang.symbols import Integer, Real
from optlang.interface import Variable
from optlang.interface.objective import Objective


EXPRESSIONS = [
    Integer(1),
    Real(1.1),
    2 * Variable("x"),
    Variable("x") + 3 * Variable("y")
]


@pytest.fixture(scope="function")
def x():
    return Variable("x")


class TestObjective(object):
    """Thoroughly test the objective class."""

    @pytest.mark.parametrize("expr", EXPRESSIONS)
    def test_init_expression(self, expr):
        Objective(expr)

    @pytest.mark.parametrize("name", [
        "R2D2",
        pytest.mark.raises("", exception=ValueError,
                           message="must not be empty"),
        pytest.mark.raises("foo bar", exception=ValueError,
                           message="cannot contain whitespace characters"),
    ])
    def test_init_name(self, x, name):
        Objective(x, name=name)

    @pytest.mark.parametrize("direction", [
        "max", "Max", "MAXIMIZE",
        "min", "Min", "MINIMIZE"
    ])
    def test_init_direction(self, x, direction):
        Objective(x, direction=direction)

    @pytest.mark.parametrize("expr, direction", [
        (Variable("x"), "max"),
        (Variable("x") + 3 * Variable("y"), "min"),
    ])
    def test_clone(self, expr, direction):
        constr = Objective(expr, direction=direction)
        new = Objective.clone(constr)
        assert new is not constr
        assert new.expression == constr.expression
        assert new.direction == constr.direction

    @pytest.mark.parametrize("kwargs, expected", [
        ({"expression": Variable("x")}, "Maximize:\n\t1.0*x"),
        ({"expression": Variable("x"), "direction": "min"},
         "Minimize:\n\t1.0*x")
    ])
    def test_dunder_str(self, kwargs, expected):
        constr = Objective(**kwargs)
        assert str(constr) == expected

    @pytest.mark.parametrize("kwargs, expected", [
        ({"expression": Variable("x")}, "<Objective 'Maximize: 1.0*x'>"),
        ({"expression": Variable("x"), "direction": "min"},
         "<Objective 'Minimize: 1.0*x'>")
    ])
    def test_dunder_repr(self, kwargs, expected):
        constr = Objective(**kwargs)
        assert repr(constr) == expected

    @pytest.mark.skip("Not implemented yet in v 2.0.")
    def test_to_dict(self):
        pass

    @pytest.mark.skip("Not implemented yet in v 2.0.")
    def test_from_dict(self):
        pass
