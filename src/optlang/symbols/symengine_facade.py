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

import symengine

Integer = symengine.Integer
Real = symengine.RealNumber
Basic = symengine.Basic
Number = symengine.Number
Zero = Integer(0)
One = Integer(1)
NegativeOne = Integer(-1)
sympify = symengine.sympify

Add = symengine.Add
Mul = symengine.Mul
Pow = symengine.Pow


class UniqueSymbol(symengine.Symbol):
    """
    Present a uniform facade to the ``symengine.Symbol`` class.

    ``symengine.Symbol`` already creates unique objects even when they have
    the same name. The use of ``__slots__`` is propagated.

    Attributes
    ----------
    name : str
        The name of the symbol.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance, the ``__slots__`` attribute is defined to be empty.
    A child class inheriting from `UniqueSymbol` is expected to define at
    least the following slots::

        __slots__ = ()

    """

    __slots__ = ()

    def __init__(self, name, **kwargs):
        super(UniqueSymbol, self).__init__(name=name, **kwargs)


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
        return One
    return Mul(*args)
