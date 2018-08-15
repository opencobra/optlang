# -*- coding: utf-8 -*-

# Copyright 2018 Novo Nordisk Foundation Center for Biosustainability,
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

from optlang.interface.change_tracker.constraint_change_tracker import (
    ConstraintChangeTracker,)
import test_expression_change_tracker as expression
import test_name_change_tracker as name
import test_bounds_change_tracker as bounds


@pytest.fixture(scope="function")
def tracker():
    return ConstraintChangeTracker()


class TestConstraintChangeTracker(expression.TestExpressionChangeTracker,
                                  name.TestNameChangeTracker,
                                  bounds.TestBoundsChangeTracker):

    def test_init(self):
        ConstraintChangeTracker()
