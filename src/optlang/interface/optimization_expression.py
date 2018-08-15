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

from __future__ import absolute_import, division

import logging
import numbers
from uuid import uuid4

from optlang.symbols import Integer, Real, Basic, sympify
from optlang.interface.variable import Variable
from optlang.interface.mixin import (
    NameMixin, SymbolicMixin, SolverStateMixin)


__all__ = ("OptimizationExpression",)

LOGGER = logging.getLogger(__name__)


class OptimizationExpression(SymbolicMixin, NameMixin, SolverStateMixin):
    """
    Define an abstract base class for Objective and Constraint.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance, the ``__slots__`` attribute is defined to be empty.
    A child class inheriting from `OptimizationExpression` is expected to
    define at least the following slots in addition to the slots of the
    parent mixins::

        __slots__ = ("_expression",)

    """

    __slots__ = ()

    def __init__(self, expression, name=None, **kwargs):
        super(OptimizationExpression, self).__init__(**kwargs)
        self._observe_symbols(expression, "expression")
        if name is None:
            self.name = str(uuid4())
        else:
            self.name = name

    def _update_expression(self):
        try:
            self._observer.update_expression(
                self, self._evaluate(self._expression))
        except (AttributeError, ReferenceError):
            # Observer is not set or no longer exists.
            pass

    def __iadd__(self, other):
        # TODO: The expression should be canonicalized.
        self._expression += other
        self._observe_symbols(other, "expression")
        self._update_expression()
        return self

    def __isub__(self, other):
        # TODO: The expression should be canonicalized.
        self._expression -= other
        self._observe_symbols(other, "expression")
        self._update_expression()
        return self

    def __imul__(self, other):
        # TODO: The expression should be canonicalized.
        self._expression *= other
        self._observe_symbols(other, "expression")
        self._update_expression()
        return self

    def __idiv__(self, other):
        # Since we imported `division`, this essentially remaps to `truediv`.
        # Mostly we handle symbolic expressions anyway, though.
        # TODO: The expression should be canonicalized.
        self._expression /= other
        self._observe_symbols(other, "expression")
        self._update_expression()
        return self

    def __itruediv__(self, other):
        # TODO: The expression should be canonicalized.
        self._expression /= other
        self._observe_symbols(other, "expression")
        self._update_expression()
        return self

    def update(self, attr):
        """Get notified about symbolic parameter value changes."""
        if attr == "expression":
            self._update_expression()
        else:
            super(OptimizationExpression, self).update(attr)

    @property
    def expression(self):
        """The mathematical expression defining the objective or constraint."""
        return self._expression

    @property
    def variables(self):
        """Variables in the constraint's or objective's expression."""
        return self.expression.atoms(Variable)

    @staticmethod
    def _canonicalize(expression):
        if isinstance(expression, numbers.Integral):
            return Integer(expression)
        elif isinstance(expression, numbers.Real):
            return Real(expression)
        else:
            # This would be a good way to canonicalize but is quite slow.
            # TODO: Re-evaluate speed with symengine.
            # expression = expression.expand()
            return expression

    @staticmethod
    def _non_linear(term):
        if len(term.atoms(Variable)) > 1:
            return True
        # What about a power of 0?
        if term.is_Pow and term.args[1] != 1:
            return True
        return False

    def is_linear(self):
        """
        Determine if the expression is linear.

        That means the expression is a polynomial of degree 0 or 1.

        """
        return any(self._non_linear(key)
                   for key in self._expression.as_coefficients_dict())

    @staticmethod
    def _is_quadratic(term):
        if len(term.atoms(Variable)) > 2:
            return True
        pass

    def is_quadratic(self):
        """
        Determine if the expression is quadratic.

        That means the expression is a polynomial with degree exactly 2.

        """
        if self._expression.is_Atom:
            return False
        if max(len(key.atoms(Variable))
               for key in self.expression.as_coefficients_dict()) == 2:
            return True
        if all((len(key.atoms(Variable)) < 2 and
                (key.is_Add or key.is_Mul or key.is_Atom)
                for key in self.expression.as_coefficients_dict())):
            return False
        if self.expression.is_Add:
            terms = self.expression.args
            is_quad = False
            for term in terms:
                if len(term.free_symbols) > 2:
                    return False
                if term.is_Pow:
                    if not term.args[1].is_Number or term.args[1] > 2:
                        return False
                    else:
                        is_quad = True
                elif term.is_Mul:
                    if len(term.free_symbols) == 2:
                        is_quad = True
                    if term.args[1].is_Pow:
                        if not term.args[1].args[1].is_Number or term.args[1].args[1] > 2:
                            return False
                        else:
                            is_quad = True
            return is_quad
        else:
            if isinstance(self.expression, Basic):
                sympy_expression = self.expression
            else:
                sympy_expression = sympify(self.expression)
            # TODO: Find a way to do this with symengine (Poly is not part of symengine, 23 March 2017)
            poly = sympy_expression.as_poly(*sympy_expression.atoms(Variable))
            if poly is None:
                return False
            else:
                return poly.is_quadratic

    def set_linear_coefficients(self, coefficients):
        """Set coefficients of linear terms in constraint or objective.
        Existing coefficients for linear or non-linear terms will not be modified.

        Note: This method interacts with the low-level solver backend and can only be used on objects that are
        associated with a Model. The method is not part of optlangs basic interface and should be used mainly where
        speed is important.

        Parameters
        ----------
        coefficients : dict
            A dictionary like {variable1: coefficient1, variable2: coefficient2, ...}

        Returns
        -------
        None
        """
        raise NotImplementedError("Child classes should implement this.")

    def get_linear_coefficients(self, variables):
        """Get coefficients of linear terms in constraint or objective.

        Note: This method interacts with the low-level solver backend and can only be used on objects that are
        associated with a Model. The method is not part of optlangs basic interface and should be used mainly where
        speed is important.

        Parameters
        ----------
        variables : iterable
            An iterable of Variable objects

        Returns
        -------
        Coefficients : dict
            {var1: coefficient, var2: coefficient ...}
        """
        raise NotImplementedError("Child classes should implement this.")
