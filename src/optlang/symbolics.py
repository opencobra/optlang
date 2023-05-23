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
This module contains a common interface to symbolic operations in Sympy and Symengine
respectively.
All symbolic operations in the optlang codebase should use these functions.
"""

from __future__ import division

import os
import six
import uuid
import logging
import optlang

logger = logging.getLogger(__name__)

# Read environment variable
SYMENGINE_PREFERENCE = os.environ.get("OPTLANG_USE_SYMENGINE", "")

if SYMENGINE_PREFERENCE.lower() in ("false", "no", "off"):
    USE_SYMENGINE = False
else:  # pragma: no cover
    try:
        import symengine
        from symengine import Symbol as symengine_Symbol
    except ImportError as e:
        if SYMENGINE_PREFERENCE.lower() in ("true", "yes", "on"):
            logger.warn("Symengine could not be imported: " + str(e))
        USE_SYMENGINE = False
    else:
        USE_SYMENGINE = True


if USE_SYMENGINE:  # pragma: no cover # noqa: C901
    optlang._USING_SYMENGINE = True

    Integer = symengine.Integer
    Real = symengine.RealDouble
    Basic = symengine.Basic
    Number = symengine.Number
    Zero = Real(0)
    One = Real(1)
    NegativeOne = Real(-1)
    sympify = symengine.sympify
    Expr = symengine.Expr

    Add = symengine.Add
    Mul = symengine.Mul
    Pow = symengine.Pow

    class Symbol(symengine_Symbol):
        """A generic symbol used in expressions."""

        def __new__(cls, name, *args, **kwargs):
            if not isinstance(name, six.string_types):
                raise TypeError("name should be a string, not %s" % repr(type(name)))

            return symengine_Symbol.__new__(cls, name)

        def __init__(self, name, *args, **kwargs):
            super(Symbol, self).__init__(name)
            self._name = name

        def __repr__(self):
            return self._name

        def __str__(self):
            return self._name

        def __getnewargs__(self):
            return (self._name, {})

    def add(*args):
        if len(args) == 1:
            args = args[0]
        elif len(args) == 0:
            return Zero
        return Add(*args)

    def mul(*args):
        if len(args) == 1:
            args = args[0]
        elif len(args) == 0:
            return One  # if you multiply nothing the result should be zero
        return Mul(*args)

else:  # Use sympy
    import sympy

    optlang._USING_SYMENGINE = False

    Integer = sympy.Integer
    Real = sympy.RealNumber
    Basic = sympy.Basic
    Number = sympy.Number
    Zero = Real(0)
    One = Real(1)
    NegativeOne = Real(-1)
    sympify = sympy.sympify
    Expr = sympy.core.expr.Expr

    Add = sympy.Add
    Mul = sympy.Mul
    Pow = sympy.Pow

    class Symbol(sympy.core.Dummy):
        """A generic symbol used in expressions."""

        def __new__(cls, name, *args, **kwargs):
            if not isinstance(name, six.string_types):
                raise TypeError("name should be a string, not %s" % repr(type(name)))

            return sympy.core.Dummy.__new__(cls, name)

        def __init__(self, *args, **kwargs):
            super(Symbol, self).__init__()

    def add(*args):
        if len(args) == 1:
            args = args[0]
        elif len(args) == 0:
            return Zero
        return sympy.Add._from_args(args)

    def mul(*args):
        if len(args) == 1:
            args = args[0]
        elif len(args) == 0:
            return One
        return sympy.Mul._from_args(args)
