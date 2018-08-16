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

import helper_symbolic_bounds as bounds
import helper_symbolic_expression as expression
from optlang.interface.constraint import Constraint


class TestConstraintSymbolicBounds(bounds.TestSymbolicBounds):
    """Test the expected behavior with integration of symbolic bounds."""

    @pytest.fixture(scope="function")
    def obj(self):
        return Constraint(1, lb=-10)


class TestConstraintSymbolicExpression(expression.TestSymbolicExpression):
    """Test the expected behavior with integration of symbolic expressions."""

    @pytest.fixture(scope="function")
    def obj(self):
        return Constraint(1, lb=-10)
