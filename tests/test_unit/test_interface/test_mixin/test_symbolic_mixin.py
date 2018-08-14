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

"""
Test the expected behavior of the `SymbolicMixin` class.

In the current implementation the `SymbolicMixin` mostly defines private
methods and cannot reasonably be tested without actual implementations such
as in `optlang.Variable` or `optlang.OptimizationExpression`.

"""

from __future__ import absolute_import

import pytest

from optlang.interface.mixin.symbolic_mixin import SymbolicMixin


class Child(SymbolicMixin):

    __slots__ = ("_foo",)

    def __init__(self, **kwargs):
        super(Child, self).__init__(**kwargs)
        self.foo = "bar"

    @property
    def foo(self):
        return self._foo

    @foo.setter
    def foo(self, value):
        self._foo = value


@pytest.fixture(scope="function")
def instance():
    return Child()


def test_init(instance):
    assert instance.foo == "bar"


def test_update(instance, mocker):
    mocked_attr = mocker.patch.object(Child, "foo",
                                      new_callable=mocker.PropertyMock,
                                      return_value="baz")
    # This line is expect to get and set the attribute, i.e., `call_count + 2`.
    instance.update("foo")
    assert instance.foo == "baz"
    assert mocked_attr.call_count == 3

