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


class TestBaseChangeTracker(object):

    @pytest.fixture(scope="function")
    def items(self, request, mocker):
        memory = {name: mocker.Mock(spec_set=["name"], return_value=name)
                  for name in request.param}
        return [memory[name] for name in request.param]

    @pytest.fixture(scope="function")
    def tuples(self, request, mocker):
        memory = {t[0]: mocker.Mock(spec_set=["name"], return_value=t[0])
                  for t in request.param}
        return [(memory[t[0]],) + t[1:] for t in request.param]

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
    def test_iter_to_add(self, tracker, items, num_unique):
        for elem in items:
            tracker.add(elem)
        res = list(tracker.iter_to_add())
        assert len(res) == num_unique
        assert set(res) == set(items)

    @pytest.mark.parametrize("items", [
        ["foo"],
        ["foo", "bar"],
        ["foo", "bar", "baz"]
    ], indirect=["items"])
    def test_remove(self, tracker, items):
        for elem in items:
            tracker.remove(elem)

    @pytest.mark.parametrize("items, num_unique", [
        (["foo"], 1),
        (["foo", "bar"], 2),
        (["foo", "bar", "baz"], 3),
        (["foo", "bar", "foo"], 2),
    ], indirect=["items"])
    def test_iter_to_remove(self, tracker, items, num_unique):
        for elem in items:
            tracker.remove(elem)
        res = list(tracker.iter_to_remove())
        assert len(res) == num_unique
        assert set(res) == set(items)
