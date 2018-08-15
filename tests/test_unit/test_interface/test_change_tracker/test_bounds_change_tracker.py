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

from optlang.interface.change_tracker.bounds_change_tracker import (
    BoundsChangeTracker,)
import test_base_change_tracker as base


@pytest.fixture(scope="function")
def tracker():
    return BoundsChangeTracker()


class TestBoundsChangeTracker(base.TestBaseChangeTracker):

    def test_init(self):
        BoundsChangeTracker()

    @pytest.mark.parametrize("tuples", [
        [("foo", 1)],
        [("foo", 1), ("bar", 2)],
        [("foo", 1), ("bar", 2), ("baz", 3)]
    ], indirect=["tuples"])
    def test_update_lb(self, tracker, tuples):
        for elem in tuples:
            tracker.update_lb(*elem)

    @pytest.mark.parametrize("tuples, unique_elements", [
        ([("foo", 1)], [0]),
        ([("foo", 1), ("bar", 2)], [0, 1]),
        ([("foo", 1), ("bar", 2), ("baz", 3)], [0, 1, 2]),
        ([("foo", 1), ("bar", 2), ("foo", 3)], [1, 2])
    ], indirect=["tuples"])
    def test_iter_lb(self, tracker, tuples, unique_elements):
        for elem in tuples:
            tracker.update_lb(*elem)
        res = list(tracker.iter_lb())
        assert len(res) == len(unique_elements)
        assert set(res) == set(tuples[i] for i in unique_elements)

    @pytest.mark.parametrize("tuples", [
        [("foo", 1)],
        [("foo", 1), ("bar", 2)],
        [("foo", 1), ("bar", 2), ("baz", 3)]
    ], indirect=["tuples"])
    def test_update_ub(self, tracker, tuples):
        for elem in tuples:
            tracker.update_ub(*elem)

    @pytest.mark.parametrize("tuples, unique_elements", [
        ([("foo", 1)], [0]),
        ([("foo", 1), ("bar", 2)], [0, 1]),
        ([("foo", 1), ("bar", 2), ("baz", 3)], [0, 1, 2]),
        ([("foo", 1), ("bar", 2), ("foo", 3)], [1, 2])
    ], indirect=["tuples"])
    def test_iter_ub(self, tracker, tuples, unique_elements):
        for elem in tuples:
            tracker.update_ub(*elem)
        res = list(tracker.iter_ub())
        assert len(res) == len(unique_elements)
        assert set(res) == set(tuples[i] for i in unique_elements)

    @pytest.mark.parametrize("tuples", [
        [("foo", 1, 3)],
        [("foo", 1, 3), ("bar", 2, 4)],
        [("foo", 1, 3), ("bar", 2, 4), ("baz", 3, 5)]
    ], indirect=["tuples"])
    def test_update_bounds(self, tracker, tuples):
        for elem in tuples:
            tracker.update_bounds(*elem)

    @pytest.mark.parametrize("tuples, unique_elements", [
        ([("foo", 1, 3)], [0]),
        ([("foo", 1, 3), ("bar", 2, 4)], [0, 1]),
        ([("foo", 1, 3), ("bar", 2, 4), ("baz", 3, 5)], [0, 1, 2]),
        ([("foo", 1, 3), ("bar", 2, 4), ("foo", 3, 5)], [1, 2])
    ], indirect=["tuples"])
    def test_iter_bounds(self, tracker, tuples, unique_elements):
        for elem in tuples:
            tracker.update_bounds(*elem)
        res = list(tracker.iter_bounds())
        assert len(res) == len(unique_elements)
        assert set(res) == set(tuples[i] for i in unique_elements)
