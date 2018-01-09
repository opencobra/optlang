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

from optlang.interface.mixins.name_mixin import NameMixin


@pytest.fixture(scope="function")
def instance():
    return NameMixin()


def test_get_name(instance):
    assert instance.name is None


@pytest.mark.parametrize("value", [
    pytest.mark.raises("", exception=ValueError, message="empty"),
    "foo",
    "foobar",
    pytest.mark.raises("foo bar", exception=ValueError, message="whitespace")
])
def test_set_name(instance, value):
    instance.name = value
    assert instance.name == value


def test_update_name(instance, mocker):
    observer = mocker.Mock(spec_set=["update_name"])
    instance.set_observer(observer)
    instance.name = "foo"
    observer.update_name.assert_called_once_with(instance, "foo")
