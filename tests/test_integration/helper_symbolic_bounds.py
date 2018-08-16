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

from optlang.interface.symbolic_parameter import SymbolicParameter


class TestSymbolicBounds(object):
    """Test the expected behavior with the integration of symbolic bounds."""

    @pytest.fixture(scope="function")
    def x(self):
        return SymbolicParameter("x")

    @pytest.fixture(scope="function")
    def y(self):
        return SymbolicParameter("y")

    def test_lb_param_observation(self, obj, x, y, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        obj.lb = 1 + x - y
        assert obj.lb == 1 + x - y
        assert mocked_attach.call_count == 2
        assert mocked_attach.call_args_list == [
            mocker.call(obj, "lb"), mocker.call(obj, "lb")]

    def test_ub_param_observation(self, obj, x, y, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        obj.ub = 1 + x - y
        assert obj.ub == 1 + x - y
        assert mocked_attach.call_count == 2
        assert mocked_attach.call_args_list == [
            mocker.call(obj, "ub"), mocker.call(obj, "ub")]

    def test_bounds_param_observation(self, obj, x, y, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        obj.bounds = (x + y, 1 + x - y)
        assert obj.bounds == (x + y, 1 + x - y)
        assert mocked_attach.call_count == 4
        assert mocked_attach.call_args_list == [
            mocker.call(obj, "bounds"), mocker.call(obj, "bounds"),
            mocker.call(obj, "bounds"), mocker.call(obj, "bounds")
        ]

    def test_lb_param_disregard(self, obj, x, y, mocker):
        mocked_detach = mocker.patch.object(SymbolicParameter, "detach")
        obj.lb = 1 + x
        obj.lb = y
        mocked_detach.assert_called_once_with(obj, "lb")

    def test_ub_param_disregard(self, obj, x, y, mocker):
        mocked_detach = mocker.patch.object(SymbolicParameter, "detach")
        obj.ub = 1 + x
        obj.ub = y
        mocked_detach.assert_called_once_with(obj, "ub")

    def test_bounds_param_disregard(self, obj, x, y, mocker):
        mocked_detach = mocker.patch.object(SymbolicParameter, "detach")
        obj.bounds = (x - y, x + y)
        obj.bounds = None, None
        assert mocked_detach.call_count == 4
        assert mocked_detach.call_args_list == [
            mocker.call(obj, "bounds"), mocker.call(obj, "bounds"),
            mocker.call(obj, "bounds"), mocker.call(obj, "bounds")
        ]
