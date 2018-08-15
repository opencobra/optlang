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

from optlang.interface.mixin.value_mixin import ValueMixin

VALUES = [-1000, -33.3, 0, None, 7.5, 100]


class Child(ValueMixin):

    __slots__ = (
        "_solver",
    )


@pytest.fixture(scope="function")
def instance():
    return Child()


def test_primal_without_observable(instance):
    assert instance.primal is None


def test_dual_without_observable(instance):
    assert instance.dual is None


@pytest.mark.parametrize("value", VALUES)
def test_primal_with_observable(instance, mocker, value):
    observable = mocker.Mock(spec_set=["get_primal"])
    observable.get_primal.return_value = value
    instance.set_solver(observable)
    assert instance.primal == value
    observable.get_primal.assert_called_once()


@pytest.mark.parametrize("value", VALUES)
def test_dual_with_observable(instance, mocker, value):
    observable = mocker.Mock(spec_set=["get_dual"])
    observable.get_dual.return_value = value
    instance.set_solver(observable)
    assert instance.dual == value
    observable.get_dual.assert_called_once()


def test_primal_with_stale_observable(instance, mocker):
    observable = mocker.Mock(spec_set=["get_primal"])
    instance.set_solver(observable)
    del observable
    assert instance.primal is None


def test_dual_with_stale_observable(instance, mocker):
    observable = mocker.Mock(spec_set=["get_dual"])
    instance.set_solver(observable)
    del observable
    assert instance.dual is None

