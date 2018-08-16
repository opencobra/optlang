# -*- coding: utf-8 -*-

# Copyright 2013-2017 Novo Nordisk Foundation Center for Biosustainability,
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
Provide a common interface to symbolic operations in sympy and symengine.

All optlang operations involving symbols must import them from here rather
than directly importing them from sympy or symengine.
"""

from __future__ import absolute_import

import os


if "OPTLANG_USE_SYMENGINE" not in os.environ:
    try:
        from optlang.symbols.symengine_facade import *
    except ImportError:
        from optlang.symbols.sympy_facade import *
elif os.environ["OPTLANG_USE_SYMENGINE"].lower() in ("true", "yes", "on", "1"):
    from optlang.symbols.symengine_facade import *
else:
    from optlang.symbols.sympy_facade import *


__all__ = (
    "Integer",
    "Real",
    "Basic",
    "Number",
    "Zero",
    "One",
    "NegativeOne",
    "sympify",
    "Add",
    "Mul",
    "Pow",
    "UniqueSymbol",
    "add",
    "mul"
)
