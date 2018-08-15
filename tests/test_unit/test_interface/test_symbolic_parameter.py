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

import weakref

import pytest

from optlang.interface.symbolic_parameter import SymbolicParameter

VALUES = [0, -10, 20, -2.3, 5.7]


@pytest.fixture(scope="function")
def parameter():
    return SymbolicParameter("simple")


@pytest.mark.parametrize("value", VALUES)
def test_init(value):
    param = SymbolicParameter("x", value)
    assert param.value == value


@pytest.mark.parametrize("value", VALUES)
def test_set_value(parameter, value):
    parameter.value = value
    assert parameter.value == value


@pytest.mark.parametrize("num", [1, 3, 9])
def test_attach_observer(mocker, parameter, num):
    observers = [mocker.Mock() for _ in range(num)]
    for o in observers:
        parameter.attach(o, "foo")
        parameter.attach(o, "bar")


@pytest.mark.parametrize("num", [1, 3, 9])
def test_notify_observer(mocker, parameter, num):
    observers = [mocker.Mock(spec_set=["update"]) for _ in range(num)]
    for o in observers:
        parameter.attach(o, "foo")
    parameter.value = 10
    for o in observers:
        o.update.assert_called_once_with("foo")


@pytest.mark.parametrize("attributes, num", [
    (["foo"], 1),
    (["foo", "bar"], 2),
    (["foo", "bar", "bar"], 2),
    (["foo", "bar", "baz"], 3)
])
def test_notify_many(mocker, parameter, attributes, num):
    observer = mocker.Mock(spec_set=["update"])
    for attr in attributes:
        parameter.attach(observer, attr)
    parameter.value = 10
    assert observer.update.call_count == num


@pytest.mark.parametrize("num", [1, 3, 9])
def test_detach_observer(mocker, parameter, num):
    observers = [mocker.Mock() for _ in range(num)]
    for o in observers:
        parameter.attach(o, "foo")
        parameter.attach(o, "bar")
    for o in observers:
        parameter.detach(o, "foo")
        parameter.detach(o, "bar")


@pytest.mark.parametrize("attributes", [
    ["foo"],
    ["foo", "bar", "baz"]
])
def test_weak_value_dict(mocker, parameter, attributes):
    observer = mocker.Mock(spec_set=["update"])
    for attr in attributes:
        parameter.attach(observer, attr)
    assert weakref.getweakrefcount(observer) == 1
    del observer
    parameter.value = 10
