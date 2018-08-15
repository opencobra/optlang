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

from optlang.interface.change_tracker.objective_change_tracker import (
    ObjectiveChangeTracker,)
import test_expression_change_tracker as expression
import test_name_change_tracker as name


@pytest.fixture(scope="function")
def tracker():
    return ObjectiveChangeTracker()


class TestObjectiveChangeTracker(expression.TestExpressionChangeTracker,
                                 name.TestNameChangeTracker):

    def test_init(self):
        ObjectiveChangeTracker()

    @pytest.mark.parametrize("tuples", [
        [("foo", "max")],
        [("foo", "max"), ("bar", "min")],
        [("foo", "max"), ("bar", "min"), ("baz", "min")]
    ], indirect=["tuples"])
    def test_update_direction(self, tracker, tuples):
        for elem in tuples:
            tracker.update_direction(*elem)

    @pytest.mark.parametrize("tuples, unique_elements", [
        ([("foo", "max")], [0]),
        ([("foo", "max"), ("bar", "min")], [0, 1]),
        ([("foo", "max"), ("bar", "min"), ("baz", "min")], [0, 1, 2]),
        ([("foo", "max"), ("bar", "min"), ("foo", "min")], [1, 2])
    ], indirect=["tuples"])
    def test_iter_direction(self, tracker, tuples, unique_elements):
        for elem in tuples:
            tracker.update_direction(*elem)
        res = list(tracker.iter_direction())
        assert len(res) == len(unique_elements)
        assert set(res) == set(tuples[i] for i in unique_elements)
