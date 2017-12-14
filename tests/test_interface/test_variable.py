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

from optlang.interface.variable import Variable

CONTINUOUS_BOUNDS = [-1023, -33.3, None, 33.3, 1023]
INTEGER_BOUNDS = [-1000, -50, None, 50, 1000]
BINARY_BOUNDS = frozenset([None, 0, 1])
BOUNDS = {
    "continuous": CONTINUOUS_BOUNDS,
    "integer": INTEGER_BOUNDS,
    "binary": BINARY_BOUNDS
}


def get_bounds_params(kind):
    bounds = BOUNDS[kind]
    return [(k,) + pair
            for k, pair in zip(repeat(kind), product(bounds, repeat=2))]


def get_bound_params(kind):
    bounds = BOUNDS[kind]
    return list(zip(repeat(kind), bounds))


def pytest_generate_tests(metafunc):
    fixtures = frozenset(metafunc.fixturenames)
    if "kind" not in fixtures:
        return
    if not hasattr(metafunc.cls, "types"):
        return
    params = list()
    if "lb" in fixtures and "ub" in fixtures:
        for kind in metafunc.cls.types:
            params.extend(get_bounds_params(kind))
        metafunc.parametrize("kind, lb, ub", params)
    elif "bound" in fixtures:
        for kind in metafunc.cls.types:
            params.extend(get_bound_params(kind))
        metafunc.parametrize("kind, bound", params)
    elif "kind" in fixtures:
        metafunc.parametrize("kind", metafunc.cls.types)


class TestVariable(object):
    """
    Thoroughly test the variable class.

    Test cases in the class are automatically parametrized depending on
    whether they rely on the ``kind`` argument (filled with the ``types``
    attribute), ``kind, bound`` arguments (filled with appropriate bound for
    the type of variable), or ``kind, lb, ub`` arguments.
    """

    types = ["continuous", "integer", "binary"]

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
def observable(mocker):
    return mocker.Mock(spec_set=["get_variable_primal", "get_variable_dual"])


class TestObservable(object):
    """Thoroughly test the get calls on the observable."""

    types = ["continuous", "integer", "binary"]

    def test_primal(self, observable, kind):
        observable.get_variable_primal.return_value = 13
        var = Variable("x", type=kind)
        var.set_observable(observable)
        assert var.primal == 13
        observable.get_variable_primal.assert_called_once_with(var)

    def test_dual(self, observable, kind):
        observable.get_variable_dual.return_value = 13
        var = Variable("x", type=kind)
        var.set_observable(observable)
        assert var.dual == 13
        observable.get_variable_dual.assert_called_once_with(var)

    def test_weakref(self, kind):
        class Observable(object):
            pass

        obj = Observable()
        var = Variable("x", type=kind)
        var.set_observable(obj)
        assert weakref.getweakrefcount(obj) == 1
        del obj
        assert var.primal is None


@pytest.fixture()
def observer(mocker):
    return mocker.Mock(spec_set=[
        "update_variable_name",
        "update_variable_type",
        "update_variable_lb",
        "update_variable_ub",
        "update_variable_bounds"
    ])


class TestObserver(object):
    """
    Thoroughly test the update calls on the observer.

    Test cases in the class are automatically parametrized depending on
    whether they rely on the ``kind`` argument (filled with the ``types``
    attribute), ``kind, bound`` arguments (filled with appropriate bound for
    the type of variable), or ``kind, lb, ub`` arguments.
    """

    types = ["continuous", "integer", "binary"]

    def test_name_update(self, observer, kind):
        old = "x"
        new = "y"
        var = Variable(old, type=kind)
        var.set_observer(observer)
        assert var.name == old
        var.name = new
        assert var.name == new
        observer.update_variable_name.assert_called_once_with(var, new)

    @pytest.mark.parametrize("old, new", list(permutations(types, 2)))
    def test_type_update(self, observer, old, new):
        var = Variable("x", type=old)
        var.set_observer(observer)
        assert var.type == old
        var.type = new
        assert var.type == new
        observer.update_variable_type.assert_called_once_with(var, new)

    def test_lb_update(self, observer, kind, bound):
        var = Variable("x", type=kind)
        var.set_observer(observer)
        assert var.lb is None
        var.lb = bound
        assert var.lb == bound
        observer.update_variable_lb.assert_called_once_with(var, bound)

    def test_ub_update(self, observer, kind, bound):
        var = Variable("x", type=kind)
        var.set_observer(observer)
        assert var.ub is None
        var.ub = bound
        assert var.ub == bound
        observer.update_variable_ub.assert_called_once_with(var, bound)

    def test_bounds_update(self, observer, kind, bound):
        var = Variable("x", type=kind)
        var.set_observer(observer)
        assert var.bounds == (None, None)
        var.bounds = bound, bound
        assert var.bounds == (bound, bound)
        observer.update_variable_bounds.assert_called_once_with(
            var, bound, bound)
