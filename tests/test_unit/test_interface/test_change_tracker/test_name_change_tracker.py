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

from optlang.interface.change_tracker.name_change_tracker import (
    NameChangeTracker,)
import test_base_change_tracker as base


@pytest.fixture(scope="function")
def tracker():
    return NameChangeTracker()


class TestNameChangeTracker(base.TestBaseChangeTracker):

    def test_init(self):
        NameChangeTracker()

    @pytest.mark.parametrize("tuples", [
        [("foo", "foo")],
        [("foo", "foo"), ("bar", "bar")],
        [("foo", "foo"), ("bar", "bar"), ("baz", "baz")]
    ], indirect=["tuples"])
    def test_update_name(self, tracker, tuples):
        for elem in tuples:
            tracker.update_name(*elem)

    @pytest.mark.parametrize("tuples, unique_elements", [
        ([("foo", "foo")], [0]),
        ([("foo", "foo"), ("bar", "bar")], [0, 1]),
        ([("foo", "foo"), ("bar", "bar"), ("baz", "baz")], [0, 1, 2]),
        ([("foo", "foo"), ("bar", "bar"), ("foo", "bar")], [1, 2])
    ], indirect=["tuples"])
    def test_iter_lb(self, tracker, tuples, unique_elements):
        for elem in tuples:
            tracker.update_name(*elem)
        res = list(tracker.iter_name())
        assert len(res) == len(unique_elements)
        assert set(res) == set(tuples[i] for i in unique_elements)
