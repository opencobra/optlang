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

from optlang.interface.variable import VariableType, Variable
from optlang.interface.symbolic_parameter import SymbolicParameter

CONTINUOUS_BOUNDS = [-1023, -33.3, None, 33.3, 1023]
INTEGER_BOUNDS = [-1000, -50, None, 50, 1000]
BINARY_BOUNDS = frozenset([None, 0, 1])
BOUNDS = {
    VariableType.CONTINUOUS: CONTINUOUS_BOUNDS,
    VariableType.INTEGER: INTEGER_BOUNDS,
    VariableType.BINARY: BINARY_BOUNDS
}


def get_separate_bounds(kind):
    bounds = BOUNDS[kind]
    return [(k,) + pair
            for k, pair in zip(repeat(kind), product(bounds, repeat=2))]


def get_bounds(kind):
    bounds = BOUNDS[kind]
    return list(zip(repeat(kind), bounds))


def pytest_generate_tests(metafunc):
    fixtures = frozenset(metafunc.fixturenames)
    if "kind" not in fixtures:
        return
    if not hasattr(metafunc.cls, "TYPES"):
        return
    params = list()
    if "lb" in fixtures and "ub" in fixtures:
        for kind in metafunc.cls.TYPES:
            params.extend(get_separate_bounds(kind))
        metafunc.parametrize("kind, lb, ub", params)
    elif "bound" in fixtures:
        for kind in metafunc.cls.TYPES:
            params.extend(get_bounds(kind))
        metafunc.parametrize("kind, bound", params)
    elif "kind" in fixtures:
        metafunc.parametrize("kind", metafunc.cls.TYPES)


class TestVariable(object):
    """
    Thoroughly test the variable class.

    Test cases in the class are automatically parametrized depending on
    whether they rely on the ``kind`` argument (filled with the ``types``
    attribute), ``kind, bound`` arguments (filled with appropriate bound for
    the type of variable), or ``kind, lb, ub`` arguments.
    """

    TYPES = list(VariableType)

    def test_init_type(self, kind):
        Variable("x", type=kind)

    @pytest.mark.parametrize("name", [
        "R2D2",
        pytest.mark.raises("", exception=ValueError,
                           message="must not be empty"),
        pytest.mark.raises("foo bar", exception=ValueError,
                           message="cannot contain whitespace characters"),
    ])
    def test_init_name(self, kind, name):
        Variable(name, type=kind)

    @pytest.mark.parametrize("var, expected", [
        (Variable("foo"), "-Inf <= foo <= Inf"),
        (Variable("foobar", lb=-10), "-10 <= foobar <= Inf"),
        (Variable("baz", ub=-10), "-Inf <= baz <= -10"),
        (Variable("baz", lb=-5, ub=5), "-5 <= baz <= 5")
    ])
    def test_dunder_str(self, var, expected):
        # Should probably introduce scientific notation in str and test floats.
        assert str(var) == expected

    @pytest.mark.parametrize("var, expected", [
        (Variable("x"), "<continuous Variable 'x'>"),
        (Variable("foo", type="binary"), "<binary Variable 'foo'>"),
        (Variable("bar", type="integer"), "<integer Variable 'bar'>"),
        (Variable("foobar", type="continuous"),
         "<continuous Variable 'foobar'>"),
    ])
    def test_dunder_repr(self, var, expected):
        assert repr(var) == expected

    def test_get_bounds(self, kind, lb, ub):
        if lb is not None and ub is not None and lb > ub:
            with pytest.raises(ValueError):
                Variable("x", type=kind, lb=lb, ub=ub)
        else:
            var = Variable("x", type=kind, lb=lb, ub=ub)
            assert var.lb == lb
            assert var.ub == ub
            assert var.bounds == (lb, ub)

    def test_set_bounds(self, kind, lb, ub):
        var = Variable("x", type=kind)
        if lb is not None and ub is not None and lb > ub:
            with pytest.raises(ValueError):
                var.bounds = lb, ub
        else:
            var.bounds = lb, ub
            assert var.lb == lb
            assert var.ub == ub
            assert var.bounds == (lb, ub)

    def test_set_lower_bound(self, kind, bound):
        var = Variable("x", type=kind, ub=0)
        if bound is not None and bound > var.ub:
            with pytest.raises(ValueError):
                var.lb = bound
        else:
            var.lb = bound
            assert var.lb == bound
            assert var.bounds == (bound, var.ub)

    def test_set_upper_bound(self, kind, bound):
        var = Variable("x", type=kind, lb=0)
        if bound is not None and var.lb > bound:
            with pytest.raises(ValueError):
                var.ub = bound
        else:
            var.ub = bound
            assert var.ub == bound
            assert var.bounds == (var.lb, bound)

    def test_primal(self, kind):
        var = Variable("x", type=kind)
        assert var.primal is None

    def test_dual(self, kind):
        var = Variable("x", type=kind)
        assert var.primal is None

    def test_clone(self, kind, bound):
        var = Variable(name="x", type=kind, lb=bound, ub=bound)
        new = Variable.clone(var)
        assert new is not var
        assert new.name == var.name
        assert new.type == var.type
        assert new.lb == var.lb
        assert new.ub == var.ub

    def test_to_dict(self, kind, bound):
        var = Variable(name="x", type=kind, lb=bound, ub=bound)
        assert var.to_dict() == {
            "name": "x",
            "type": kind,
            "lb": bound,
            "ub": bound
        }

    def test_from_dict(self, kind, bound):
        var = Variable.from_dict({
            "name": "x",
            "type": kind,
            "lb": bound,
            "ub": bound
        })
        assert var.name == "x"
        assert var.type == kind
        assert var.lb == bound
        assert var.ub == bound


@pytest.mark.parametrize("lb, ub", [
    (-5, 5),
    (3.3, 5.3)
])
def test_non_binary_bounds(lb, ub):
    with pytest.raises(ValueError, message="Binary variable's bounds"):
        Variable("x", type="binary", lb=lb, ub=ub)


@pytest.fixture()
def solver(mocker):
    return mocker.Mock(spec_set=["get_primal", "get_dual"])


class TestSolverState(object):
    """Thoroughly test the get calls on the solver state reference."""

    TYPES = list(VariableType)

    def test_primal(self, solver, kind):
        solver.get_primal.return_value = 13
        var = Variable("x", type=kind)
        var.set_solver(solver)
        assert var.primal == 13
        solver.get_primal.assert_called_once_with(var)

    def test_dual(self, solver, kind):
        solver.get_dual.return_value = 13
        var = Variable("x", type=kind)
        var.set_solver(solver)
        assert var.dual == 13
        solver.get_dual.assert_called_once_with(var)

    def test_weakref(self, kind):

        class SolverState(object):
            pass

        obj = SolverState()
        var = Variable("x", type=kind)
        var.set_solver(obj)
        assert weakref.getweakrefcount(obj) == 1
        del obj
        assert var.primal is None


@pytest.fixture()
def observer(mocker):
    return mocker.Mock(spec_set=[
        "update_name",
        "update_type",
        "update_lb",
        "update_ub",
        "update_bounds"
    ])


class TestObserver(object):
    """
    Thoroughly test the update calls on the observer.

    Test cases in the class are automatically parametrized depending on
    whether they rely on the ``kind`` argument (filled with the ``types``
    attribute), ``kind, bound`` arguments (filled with appropriate bound for
    the type of variable), or ``kind, lb, ub`` arguments.
    """

    TYPES = list(VariableType)

    def test_name_update(self, observer, kind):
        old = "x"
        new = "y"
        var = Variable(old, type=kind)
        var.subscribe(observer)
        assert var.name == old
        var.name = new
        assert var.name == new
        observer.update_name.assert_called_once_with(var, new)

    @pytest.mark.parametrize("old, new", list(permutations(TYPES, 2)))
    def test_type_update(self, observer, old, new):
        var = Variable("x", type=old)
        var.subscribe(observer)
        assert var.type == old
        var.type = new
        assert var.type == new
        observer.update_type.assert_called_once_with(var, new)

    def test_lb_update(self, observer, kind, bound):
        var = Variable("x", type=kind)
        var.subscribe(observer)
        assert var.lb is None
        var.lb = bound
        assert var.lb == bound
        observer.update_lb.assert_called_once_with(var, bound)

    def test_ub_update(self, observer, kind, bound):
        var = Variable("x", type=kind)
        var.subscribe(observer)
        assert var.ub is None
        var.ub = bound
        assert var.ub == bound
        observer.update_ub.assert_called_once_with(var, bound)

    def test_bounds_update(self, observer, kind, bound):
        var = Variable("x", type=kind)
        var.subscribe(observer)
        assert var.bounds == (None, None)
        var.bounds = bound, bound
        assert var.bounds == (bound, bound)
        observer.update_bounds.assert_called_once_with(
            var, bound, bound)


@pytest.fixture(scope="function")
def x():
    return SymbolicParameter("x")


@pytest.fixture(scope="function")
def y():
    return SymbolicParameter("y")


class TestSymbolicBounds(object):
    """
    Test the expected behavior with integration of symbolic bounds.

    Since we use ``__slots__``, the instance attributes and methods cannot be
    mocked directly. They are read-only but mock tries to add and remove the
    attribute or method. Hence we mock the class definition.

    """

    TYPES = list(VariableType)

    def test_lb_param_observation(self, x, y, kind, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        var = Variable("i", type=kind)
        var.lb = 1 + x - y
        assert var.lb == 1 + x - y
        assert mocked_attach.call_count == 2
        assert mocked_attach.call_args_list == [
            mocker.call(var, "lb"), mocker.call(var, "lb")]

    def test_ub_param_observation(self, x, y, kind, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        var = Variable("i", type=kind)
        var.ub = 1 + x - y
        assert var.ub == 1 + x - y
        assert mocked_attach.call_count == 2
        assert mocked_attach.call_args_list == [
            mocker.call(var, "ub"), mocker.call(var, "ub")]

    def test_bounds_param_observation(self, x, y, kind, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        var = Variable("i", type=kind)
        var.bounds = (x + y, 1 + x - y)
        assert var.bounds == (x + y, 1 + x - y)
        assert mocked_attach.call_count == 4
        assert mocked_attach.call_args_list == [
            mocker.call(var, "bounds"), mocker.call(var, "bounds"),
            mocker.call(var, "bounds"), mocker.call(var, "bounds")
        ]

    def test_lb_param_disregard(self, x, y, kind, mocker):
        mocked_detach = mocker.patch.object(SymbolicParameter, "detach")
        var = Variable("i", type=kind)
        var.lb = 1 + x
        var.lb = y
        mocked_detach.assert_called_once_with(var, "lb")

    def test_ub_param_disregard(self, x, y, kind, mocker):
        mocked_detach = mocker.patch.object(SymbolicParameter, "detach")
        var = Variable("i", type=kind)
        var.ub = 1 + x
        var.ub = y
        mocked_detach.assert_called_once_with(var, "ub")

    def test_bounds_param_disregard(self, x, y, kind, mocker):
        mocked_detach = mocker.patch.object(SymbolicParameter, "detach")
        var = Variable("i", type=kind)
        var.bounds = (x - y, x + y)
        var.bounds = None, None
        assert mocked_detach.call_count == 4
        assert mocked_detach.call_args_list == [
            mocker.call(var, "bounds"), mocker.call(var, "bounds"),
            mocker.call(var, "bounds"), mocker.call(var, "bounds")
        ]
