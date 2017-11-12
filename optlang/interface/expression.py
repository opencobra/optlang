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

from __future__ import absolute_import

__all__ = ("OptimizationExpression",)


# noinspection PyPep8Naming
class OptimizationExpression(object):
    """Abstract base class for Objective and Constraint."""

    @classmethod
    def _substitute_variables(cls, expression, model=None, **kwargs):
        """Substitutes variables in (optimization)expression (constraint/objective) with variables of the appropriate interface type.
        Attributes
        ----------
        expression: Constraint, Objective
            An optimization expression.
        model: Model or None, optional
            A reference to an optimization model that should be searched for appropriate variables first.
        """
        interface = sys.modules[cls.__module__]
        variable_substitutions = dict()
        for variable in expression.variables:
            if model is not None and variable.name in model.variables:
                # print(variable.name, id(variable.problem))
                variable_substitutions[variable] = model.variables[variable.name]
            else:
                variable_substitutions[variable] = interface.Variable.clone(variable)
        adjusted_expression = expression.expression.xreplace(variable_substitutions)
        return adjusted_expression

    def __init__(self, expression, name=None, problem=None, sloppy=False, *args, **kwargs):
        # Ensure that name is str and not binary of unicode - some solvers only support string type in Python 2.
        if six.PY2 and name is not None:
            name = str(name)

        super(OptimizationExpression, self).__init__(*args, **kwargs)
        self._problem = problem
        if sloppy:
            self._expression = expression
        else:
            self._expression = self._canonicalize(expression)
        if name is None:
            self._name = str(uuid.uuid1())
        else:
            self._name = name

    @property
    def name(self):
        """The name of the object"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def problem(self):
        """A reference to the model that the object belongs to (or None)"""
        return getattr(self, '_problem', None)

    @problem.setter
    def problem(self, value):
        self._problem = value

    def _get_expression(self):
        return self._expression

    @property
    def expression(self):
        """The mathematical expression defining the objective/constraint."""
        return self._get_expression()

    @property
    def variables(self):
        """Variables in constraint/objective's expression."""
        return self.expression.atoms(Variable)

    def _canonicalize(self, expression):
        if isinstance(expression, float):
            return symbolics.Real(expression)
        elif isinstance(expression, int):
            return symbolics.Integer(expression)
        else:
            # expression = expression.expand() This would be a good way to canonicalize, but is quite slow
            return expression

    @property
    def is_Linear(self):
        """Returns True if expression is linear (a polynomial with degree 1 or 0) (read-only)."""
        coeff_dict = self.expression.as_coefficients_dict()
        for key in coeff_dict.keys():
            if len(key.free_symbols) < 2 and (key.is_Add or key.is_Mul or key.is_Atom):
                pass
            else:
                return False
            if key.is_Pow and key.args[1] != 1:
                return False
        else:
            return True

    @property
    def is_Quadratic(self):
        """Returns True if the expression is a polynomial with degree exactly 2 (read-only)."""
        if self.expression.is_Atom:
            return False
        if all((len(key.free_symbols) < 2 and (key.is_Add or key.is_Mul or key.is_Atom)
                for key in self.expression.as_coefficients_dict().keys())):
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
            if isinstance(self.expression, sympy.Basic):
                sympy_expression = self.expression
            else:
                sympy_expression = sympy.sympify(self.expression)
            # TODO: Find a way to do this with symengine (Poly is not part of symengine, 23 March 2017)
            poly = sympy_expression.as_poly(*sympy_expression.atoms(sympy.Symbol))
            if poly is None:
                return False
            else:
                return poly.is_quadratic

    def __iadd__(self, other):
        self._expression += other
        return self

    def __isub__(self, other):
        self._expression -= other
        return self

    def __imul__(self, other):
        self._expression *= other
        return self

    def __idiv__(self, other):
        self._expression /= other
        return self

    def __itruediv__(self, other):
        self._expression /= other
        return self

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

