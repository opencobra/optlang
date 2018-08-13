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

from optlang.interface.mixins.solver_state_mixin import SolverStateMixin


class Child(SolverStateMixin):

    __slots__ = ("_solver",)


@pytest.fixture(scope="function")
def instance():
    return Child()


def test_set_solver(instance, mocker):
    solver = mocker.Mock()
    instance.set_solver(solver)
    assert weakref.getweakrefcount(solver) == 1
    second = mocker.Mock()
    instance.set_solver(second)
    assert weakref.getweakrefcount(solver) == 0


def test_unset_solver(instance, mocker):
    solver = mocker.Mock()
    instance.set_solver(solver)
    assert weakref.getweakrefcount(solver) == 1
    instance.unset_solver()
    assert weakref.getweakrefcount(solver) == 0
