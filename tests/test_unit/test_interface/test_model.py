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

from optlang.symbols import Zero
from optlang.interface.constraint import Constraint
from optlang.interface.objective import Objective
from optlang.interface.variable import Variable
from optlang.interface.model import Model


class TestModel(object):
    """Thoroughly test the constraint class."""

    @pytest.fixture(scope="function")
    def model(self):
        return Model()

    def test_init(self):
        Model()

    @pytest.mark.parametrize("name", [
        None,
        "R2D2",
        "",
        "foo bar"
    ])
    def test_init_name(self, name):
        Model(name=name)

    @pytest.mark.parametrize("objective", [
        None,
        Objective(Zero),
        pytest.mark.raises(1, exception=TypeError, message="not <class 'int'>")
    ])
    def test_init_objective(self, objective):
        Model(objective=objective)

    @pytest.mark.parametrize("variables", [
        None,
        Variable("x"),
        [Variable("x"), Variable("y")],
        [Variable("x"), Variable("y"), Variable("z")]
    ])
    def test_init_variables(self, variables):
        Model(variables=variables)

    @pytest.mark.parametrize("constraints", [
        None,
        Constraint(Zero, lb=-1),
        [Constraint(Zero, lb=-1), Constraint(Zero, lb=-1)],
        [Constraint(Zero, lb=-1), Constraint(Zero, lb=-1),
         Constraint(Zero, lb=-1)]
    ])
    def test_init_constraints(self, constraints):
        Model(constraints=constraints)

    @pytest.mark.parametrize("elements", [
        Variable("x"),
        Constraint(Zero, lb=-1),
        Objective(Zero),
        [],
        [Variable("x")],
        [Constraint(Zero, lb=-1)],
        [Objective(Zero)],
        [Variable("x"), Variable("y")],
        [Constraint(Zero, lb=-1), Constraint(Zero, lb=-1)],
        [Objective(Zero), Objective(Zero)],
    ])
    def test_add(self, model, elements):
        model.add(elements)

    def test_add_sloppy(self, model, elements):
        model.add(elements, sloppy=True)

    def test_remove(self, model):
        assert False

    def test_update(self, model, mocker):
        mocker.spy(model, "update")
        var = Variable("x")
        model.add(var)
        model.remove(var)
        assert model.update.call_count == 1

    @pytest.mark.raises(NotImplementedError, message="high level interface")
    def test_optimize(self, model):
        model.optimize()

