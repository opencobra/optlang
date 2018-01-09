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

from optlang.interface.symbolic_parameter import SymbolicParameter

__all__ = ("SymbolicMixin",)


class SymbolicMixin(object):
    """Provide methods for handling symbolic parameters."""

    def __init__(self, **kwargs):
        super(SymbolicMixin, self).__init__(**kwargs)

    def _observe_symbols(self, expression, attr):
        try:
            for sym in expression.atoms(SymbolicParameter):
                sym.attach(self, attr)
        except AttributeError:
            pass

    def _disregard_symbols(self, expression, attr):
        try:
            for sym in expression.atoms(SymbolicParameter):
                sym.detach(self, attr)
        except AttributeError:
            pass

    def notify(self, attr):
        """Notify an attribute of a symbolic parameter value change."""
        # For now this is an easy way to trigger all the needed updates.
        setattr(self, attr, getattr(self, attr))

    @staticmethod
    def _evaluate(expression):
        try:
            return expression.subs({
                sym: sym.value for sym in expression.atoms(SymbolicParameter)})
        except AttributeError:
            return expression
