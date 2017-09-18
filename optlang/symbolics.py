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
This module contains a common interface to symbolic operations in Sympy and Symengine respectively.
All symbolic operations in the optlang codebase should use these functions.
"""

from __future__ import division

import os
import six
import uuid
import logging

logger = logging.getLogger(__name__)

# Read environment variable
SYMENGINE_PREFERENCE = os.environ.get("OPTLANG_USE_SYMENGINE", "")

if SYMENGINE_PREFERENCE.lower() in ("false", "no", "off"):
    USE_SYMENGINE = False
else:  # pragma: no cover
    try:
        import symengine
        import symengine.sympy_compat
        from symengine.sympy_compat import Symbol as symengine_Symbol
    except ImportError as e:
        if SYMENGINE_PREFERENCE.lower() in ("true", "yes", "on"):
            logger.warn("Symengine could not be imported: " + str(e))
        USE_SYMENGINE = False
    else:
        USE_SYMENGINE = True
        logger.warn(
            "Loading symengine... This feature is in beta testing. " +
            "Please report any issues you encounter on http://github.com/biosustain/optlang/issues"
        )


if USE_SYMENGINE:  # pragma: no cover
    import operator
    from six.moves import reduce

    Integer = symengine.Integer
    Real = symengine.RealDouble
    Zero = Integer(0)
    One = Integer(1)
    NegativeOne = Integer(-1)
    sympify = symengine.sympy_compat.sympify

    Add = symengine.Add
    Mul = symengine.Mul
    Pow = symengine.sympy_compat.Pow

    class Symbol(symengine_Symbol):
        def __new__(cls, name, *args, **kwargs):
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
        return sum(args)

    def mul(*args):
        if len(args) == 1:
            args = args[0]
        return reduce(operator.mul, args, 1)

else:  # Use sympy
    import sympy
    from sympy.core.assumptions import _assume_rules
    from sympy.core.facts import FactKB
    from sympy.core.expr import Expr

    Integer = sympy.Integer
    Real = sympy.RealNumber
    Zero = Integer(0)
    One = Integer(1)
    NegativeOne = Integer(-1)
    sympify = sympy.sympify

    Add = sympy.Add
    Mul = sympy.Mul
    Pow = sympy.Pow

    class Symbol(sympy.Symbol):
        def __new__(cls, name, **kwargs):
            if not isinstance(name, six.string_types):
                raise TypeError("name should be a string, not %s" % repr(type(name)))

            obj = sympy.Symbol.__new__(cls, str(uuid.uuid1()))

            obj.name = name
            obj._assumptions = FactKB(_assume_rules)
            obj._assumptions._tell('commutative', True)
            obj._assumptions._tell('uuid', uuid.uuid1())

            return obj

        def __init__(self, *args, **kwargs):
            super(Symbol, self).__init__()

    def add(*args):
        if len(args) == 1:
            args = args[0]
        return sympy.Add._from_args(args)

    def mul(*args):
        if len(args) == 1:
            args = args[0]
        return sympy.Mul._from_args(args)
