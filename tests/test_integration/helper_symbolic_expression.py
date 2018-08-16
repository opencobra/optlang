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
from optlang.interface.variable import Variable


class TestSymbolicExpression(object):
    """Test the expected behavior with the integration of symbolic bounds."""

    @pytest.fixture(scope="function")
    def expr(self):
        return SymbolicParameter("mu") + Variable("x") + SymbolicParameter(
            "rho") * Variable("y")

    def test_expression_observation(self, obj, expr, mocker):
        mocked_attach = mocker.patch.object(SymbolicParameter, "attach")
        obj.expression = expr
        assert obj.expression == expr
        assert mocked_attach.call_count == 2
        assert mocked_attach.call_args_list == [
            mocker.call(obj, "expression"), mocker.call(obj, "expression")]

