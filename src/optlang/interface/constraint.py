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

from optlang.interface.optimization_expression import OptimizationExpression
from optlang.interface.mixins import BoundsMixin, ValueMixin

__all__ = ("Constraint",)


class Constraint(OptimizationExpression, BoundsMixin, ValueMixin):
    """
    Constraint objects represent the mathematical (in-)equalities that constrain
    an optimization problem.  A constraint is formulated by a symbolic
    expression of variables and a lower and/or upper bound.  Equality
    constraints can be formulated by setting the upper and lower bounds to the
    same value.

    Some solvers support indicator variables. This lets a binary variable act as
    a switch that decides whether the constraint should be active (cannot be
    violated) or inactive (can be violated).

    The constraint expression can be an arbitrary combination of variables,
    however the individual solvers have limits to the forms of constraints they
    allow. Most solvers only allow linear constraints, meaning that
    the expression should be of the form::

        a * var1 + b * var2 + c * var3 ...

    Attributes
    ----------
    expression: sympy or symengine
        The mathematical expression defining the constraint.
    name: str, optional
        The constraint's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    indicator_variable: Variable
        The indicator variable (needs to be binary).
    active_when: 0 or 1 (default 0)
        When the constraint should

    Examples
    ----------
    >>> expr = 2.4 * var1 - 3.8 * var2
    >>> c1 = Constraint(expr, lb=0, ub=10)

    >>> indicator_var = Variable("var3", type="binary") # Only possible with some solvers
    >>> c2 = Constraint(var2, lb=0, ub=0, indicator_variable=indicator_var, active_when=1) # When the indicator is 1, var2 is constrained to be 0

    """

    __slots__ = (
        "_observer",
        "_solver",
        "_name",
        "_lb", "_numeric_lb", "_ub", "_numeric_ub",
        "_expression",
        "_indicator_variable",
        "_active_when",
        "__weakref__"
    )

    _INDICATOR_CONSTRAINT_SUPPORT = True

    def __init__(self, expression, lb=None, ub=None, sloppy=False,
                 indicator_variable=None, active_when=1, **kwargs):
        """
        Initialize a constraint with an expression.

        Parameters
        ----------
        expression: sympy or symengine
            The mathematical expression defining the constraint.
        name: str, optional
            The constraint's name.
        lb: float or None, optional
            The lower bound, if None then -inf.
        ub: float or None, optional
            The upper bound, if None then inf.
        indicator_variable: Variable
            The indicator variable (needs to be binary).
        active_when: 0 or 1 (default 0)
            When the constraint should

        """
        super(Constraint, self).__init__(expression=expression, **kwargs)
        self.bounds = lb, ub
        if sloppy:
            self._expression = expression
        else:
            self._expression = self._canonicalize(expression)
        self.__check_valid_indicator_variable(indicator_variable)
        self.__check_valid_active_when(active_when)
        self._indicator_variable = indicator_variable
        self._active_when = active_when

    @classmethod
    def __check_valid_indicator_variable(cls, variable):
        if variable is not None and not cls._INDICATOR_CONSTRAINT_SUPPORT:
            raise IndicatorConstraintsNotSupported(
                'Solver interface %s does not support indicator constraints' % cls.__module__)
        if variable is not None and variable.type != 'binary':
            raise ValueError('Provided indicator variable %s is not binary.' % variable)

    @staticmethod
    def __check_valid_active_when(active_when):
        if active_when != 0 and active_when != 1:
            raise ValueError('Provided active_when argument %s needs to be either 1 or 0' % active_when)

    @classmethod
    def clone(cls, constraint, **kwargs):
        """
        Copy the attributes of another constraint.

        Parameters
        ----------
        constraint: optlang.Constraint (or subclass)
            The constraint to copy.

        Example
        ----------
        >>> const_copy = Constraint.clone(old_constraint)
        """
        return cls(
            expression=constraint.expression,
            lb=constraint.lb, ub=constraint.ub,
            indicator_variable=constraint.indicator_variable,
            active_when=constraint.active_when,
            name=constraint.name, sloppy=True, **kwargs)

    @property
    def indicator_variable(self):
        """The indicator variable of constraint (if available)."""
        return self._indicator_variable

    # @indicator_variable.setter
    # def indicator_variable(self, value):
    #     self.__check_valid_indicator_variable(value)
    #     self._indicator_variable = value

    @property
    def active_when(self):
        """Activity relation of constraint to indicator variable (if supported)."""
        return self._active_when

    def __str__(self):
        if self.lb is not None:
            lhs = str(self.lb) + ' <= '
        else:
            lhs = ''
        if self.ub is not None:
            rhs = ' <= ' + str(self.ub)
        else:
            rhs = ''
        if self.indicator_variable is not None:
            lhs = self.indicator_variable.name + ' = ' + str(self.active_when) + ' -> ' + lhs
        return str(self.name) + ": " + lhs + self.expression.__str__() + rhs

    def _canonicalize(self, expression):
        expression = super(Constraint, self)._canonicalize(expression)
        if expression.is_Atom or expression.is_Mul:
            return expression
        lonely_coeffs = [arg for arg in expression.args if arg.is_Number]
        if not lonely_coeffs:
            return expression
        assert len(lonely_coeffs) == 1
        coeff = lonely_coeffs[0]
        expression = expression - coeff
        if self.lb is not None and self.ub is not None:
            oldub = self.ub
            self.ub = None
            self.lb = self.lb - float(coeff)
            self.ub = oldub - float(coeff)
        elif self.lb is not None:
            self.lb = self.lb - float(coeff)
        elif self.ub is not None:
            self.ub = self.ub - float(coeff)
        else:
            raise ValueError(
                "{} cannot be shaped into canonical form if neither lower or "
                "upper constraint bounds are set.".format(expression)
            )
        return expression

    def _round_primal_to_bounds(self, primal, tolerance=1e-5):
        if (self.lb is None or primal >= self.lb) and (self.ub is None or primal <= self.ub):
            return primal
        else:
            if (primal <= self.lb) and ((self.lb - primal) <= tolerance):
                return self.lb
            elif (primal >= self.ub) and ((self.ub - primal) >= -tolerance):
                return self.ub
            else:
                raise AssertionError(
                    'The primal value %s returned by the solver is out of bounds for variable %s (lb=%s, ub=%s)' % (
                        primal, self.name, self.lb, self.ub))

    def to_dict(self):
        """
        Returns a json-compatible object from the constraint that can be saved using the json module.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(constraint.to_json(), outfile)
        """
        if self.indicator_variable is None:
            indicator = None
        else:
            indicator = self.indicator_variable.name
        json_obj = {
            "name": self.name,
            "expression": expr_to_json(self.expression),
            "lb": self.lb,
            "ub": self.ub,
            "indicator_variable": indicator,
            "active_when": self.active_when
        }
        return json_obj

    @classmethod
    def from_dict(cls, json_obj, variables=None):
        """
        Constructs a Variable from the provided json-object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     constraint = Constraint.from_json(json.load(infile))
        """
        if variables is None:
            variables = {}
        expression = parse_expr(json_obj["expression"], variables)
        if json_obj["indicator_variable"] is None:
            indicator = None
        else:
            indicator = variables[json_obj["indicator_variable"]]
        return cls(
            expression,
            name=json_obj["name"],
            lb=json_obj["lb"],
            ub=json_obj["ub"],
            indicator_variable=indicator,
            active_when=json_obj["active_when"]
        )

