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

from optlang.symbolics import Symbol


class SymbolicParameter(Symbol):
    """
    A symbolic parameter to be used in bounds and constraints.

    Attributes
    ----------
    value : numeric
        The current numeric value of the parameter. Changing the value will
        update all expression this parameter is part of.
    """

    def __init__(self, name, value=0, **kwargs):
        super(SymbolicParameter, self).__init__(name=name, **kwargs)
        self._observers = set()
        self._value = value

    @property
    def value(self):
        """Return the associated numeric value."""
        return self._value

    @value.setter
    def value(self, other):
        """Set a new value and update all expressions."""
        self._value = other
        for obs in self._observers:
            obs.

    def register(self, assigned, attr):
        """Register an object and its expression with this instance."""
        self._observers.add((assigned, attr))

    def unregister(self, assigned):
        """Unregister an object from this instance."""
        self._observers.remove(None)

    @staticmethod
    def handle_symbols(expression, instance, attr):
        try:
            for sym in expression.atoms(SymbolicParameter):
                sym.register(instance, attr)
            return SymbolicExpressionWrapper(expression)
        except AttributeError:
            return expression

    @staticmethod
    def evaluate(expression):
        return expression.subs([
            (sym, sym.value) for sym in expression.atoms(SymbolicParameter)])
