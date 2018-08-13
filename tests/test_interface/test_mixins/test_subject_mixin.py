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

from optlang.interface.mixins.subject_mixin import SubjectMixin


class Child(SubjectMixin):

    __slots__ = ("_observer",)


@pytest.fixture(scope="function")
def instance():
    return Child()


def test_subscribe(instance, mocker):
    observer = mocker.Mock()
    instance.subscribe(observer)
    assert weakref.getweakrefcount(observer) == 1
    second = mocker.Mock()
    instance.subscribe(second)
    assert weakref.getweakrefcount(observer) == 0


def test_unsubscribe(instance, mocker):
    observer = mocker.Mock()
    instance.subscribe(observer)
    assert weakref.getweakrefcount(observer) == 1
    instance.unsubscribe()
    assert weakref.getweakrefcount(observer) == 0
