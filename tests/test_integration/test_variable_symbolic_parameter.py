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

import helper_symbolic_bounds as bounds
from optlang.interface.variable import VariableType, Variable


def pytest_generate_tests(metafunc):
    fixtures = frozenset(metafunc.fixturenames)
    if "obj" not in fixtures:
        return
    if not hasattr(metafunc.cls, "TYPES"):
        return
    if "obj" in fixtures:
        metafunc.parametrize("obj", [Variable("i", type=t)
                                     for t in metafunc.cls.TYPES])


class TestVariableSymbolicBounds(bounds.TestSymbolicBounds):
    """
    Test the expected behavior with integration of symbolic bounds.

    Since we use ``__slots__``, the instance attributes and methods cannot be
    mocked directly. They are read-only but mock tries to add and remove the
    attribute or method. Hence we mock the class definition.

    """

    TYPES = list(VariableType)


