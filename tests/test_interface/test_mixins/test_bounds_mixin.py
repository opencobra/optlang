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

from optlang.interface.mixins.bounds_mixin import BoundsMixin

VALUES = [-1000, -33.3, 0, None, 7.5, 100]


class Child(BoundsMixin):

    __slots__ = (
        "_observer",
        "_lb", "_numeric_lb", "_ub", "_numeric_ub"
    )


@pytest.fixture(scope="function")
def instance():
    return Child()


def test_init(instance):
    assert instance.lb is None
    assert instance.ub is None
    assert instance.bounds == (None, None)


@pytest.mark.parametrize("value", VALUES)
def test_numeric_lb(instance, value):
    instance.lb = value
    assert instance.lb == value


@pytest.mark.parametrize("value", VALUES)
def test_numeric_ub(instance, value):
    instance.ub = value
    assert instance.ub == value


@pytest.mark.parametrize("value", VALUES)
def test_numeric_bounds(instance, value):
    instance.bounds = value, value
    assert instance.bounds == (value, value)


@pytest.mark.parametrize("value", [
    -10,
    None,
    pytest.mark.raises(10, exception=ValueError, message="less than")
])
def test_check_numeric_lb(instance, value):
    instance.ub = 0
    instance.lb = value


@pytest.mark.parametrize("value", [
    pytest.mark.raises(-10, exception=ValueError, message="less than"),
    None,
    10,
])
def test_check_numeric_ub(instance, value):
    instance.lb = 0
    instance.ub = value


@pytest.mark.parametrize("lb, ub", [
    pytest.mark.raises((10, -10), exception=ValueError, message="less than"),
    (None, -100),
    (100, None),
    (None, None)
])
def test_check_numeric_bounds(instance, lb, ub):
    instance.bounds = lb, ub


def test_update_numeric_lb(instance, mocker):
    observer = mocker.Mock(spec_set=["update_lb"])
    instance.subscribe(observer)
    instance.lb = 10
    observer.update_lb.assert_called_once_with(instance, 10)


def test_update_numeric_ub(instance, mocker):
    observer = mocker.Mock(spec_set=["update_ub"])
    instance.subscribe(observer)
    instance.ub = 10
    observer.update_ub.assert_called_once_with(instance, 10)


def test_update_numeric_bounds(instance, mocker):
    observer = mocker.Mock(spec_set=["update_bounds"])
    instance.subscribe(observer)
    instance.bounds = 10, 10
    observer.update_bounds.assert_called_once_with(instance, 10, 10)
