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

from optlang.interface.change_tracker.base_change_tracker import (
    BaseChangeTracker,)


@pytest.fixture(scope="function")
def tracker():
    return BaseChangeTracker()


@pytest.fixture(scope="function")
def items(request, mocker):
    return [mocker.Mock(spec_set=["name"], return_value=name)
            for name in request.param]


class TestBaseChangeTracker(object):

    def test_init(self):
        BaseChangeTracker()

    @pytest.mark.parametrize("items", [
        ["foo"],
        ["foo", "bar"],
        ["foo", "bar", "baz"]
    ], indirect=["items"])
    def test_add(self, tracker, items):
        for elem in items:
            tracker.add(elem)

    @pytest.mark.parametrize("items, num_unique", [
        (["foo"], 1),
        (["foo", "bar"], 2),
        (["foo", "bar", "baz"], 3),
        (["foo", "bar", "foo"], 2),
    ], indirect=["items"])
    def test_lazy_add(self, tracker, items, num_unique):
        # TODO: Mocks are not unique by name thus this fails.
        for elem in items:
            tracker.add(elem)
        res = list(tracker.iter_to_add())
        assert len(res) == num_unique
        assert set(res) == set(items)
