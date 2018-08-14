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

from optlang.interface.variable import VariableType, Variable
from optlang.interface.symbolic_parameter import SymbolicParameter


def pytest_generate_tests(metafunc):
    fixtures = frozenset(metafunc.fixturenames)
    if "kind" not in fixtures:
        return
    if not hasattr(metafunc.cls, "TYPES"):
        return
    if "kind" in fixtures:
        metafunc.parametrize("kind", metafunc.cls.TYPES)


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
