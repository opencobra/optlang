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

from optlang.interface.change_tracker.variable_change_tracker import (
    VariableChangeTracker,)
import test_name_change_tracker as name
import test_bounds_change_tracker as bounds


@pytest.fixture(scope="function")
def tracker():
    return VariableChangeTracker()


class TestConstraintChangeTracker(name.TestNameChangeTracker,
                                  bounds.TestBoundsChangeTracker):

    def test_init(self):
        VariableChangeTracker()

    @pytest.mark.parametrize("tuples", [
        [("foo", "continuous")],
        [("foo", "continuous"), ("bar", "integer")],
        [("foo", "continuous"), ("bar", "integer"), ("baz", "binary")]
    ], indirect=["tuples"])
    def test_update_type(self, tracker, tuples):
        for elem in tuples:
            tracker.update_type(*elem)

    @pytest.mark.parametrize("tuples, unique_elements", [
        ([("foo", "continuous")], [0]),
        ([("foo", "continuous"), ("bar", "integer")], [0, 1]),
        ([("foo", "continuous"), ("bar", "integer"), ("baz", "binary")],
         [0, 1, 2]),
        ([("foo", "continuous"), ("bar", "integer"), ("foo", "binary")],
         [1, 2])
    ], indirect=["tuples"])
    def test_iter_type(self, tracker, tuples, unique_elements):
        for elem in tuples:
            tracker.update_type(*elem)
        res = list(tracker.iter_type())
        assert len(res) == len(unique_elements)
        assert set(res) == set(tuples[i] for i in unique_elements)
