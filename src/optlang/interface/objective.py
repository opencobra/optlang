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

from optlang.symbols import Basic
from optlang.interface.optimization_expression import OptimizationExpression

__all__ = ("Objective",)


class Objective(OptimizationExpression):
    """
    Objective objects are used to represent the objective function of an optimization problem.
    An objective consists of a symbolic expression of variables in the problem and a direction. The direction
    can be either 'min' or 'max' and specifies whether the problem is a minimization or a maximization _problem.

    After a problem has been optimized, the optimal objective value can be accessed from the 'value' attribute
    of the model's objective, i.e. :code:`obj_val = model.objective.value`.

    Attributes
    ----------
    expression: sympy
        The mathematical expression defining the objective.
    name: str, optional
        The name of the constraint.
    direction: 'max' or 'min'
        The optimization direction.
    value: float, read-only
        The current objective value.
    problem: solver
        The low-level solver object.

    """

    __slots__ = (
        "_observer",
        "_solver",
        "_name",
        "_expression",
        "_direction",
        "__weakref__"
    )

    def __init__(self, expression, direction='max', sloppy=False,
                 **kwargs):
        super(Objective, self).__init__(expression=expression, **kwargs)
        if sloppy:
            self._expression = expression
        else:
            self._expression = self._canonicalize(expression)
        self._direction = None
        self.direction = direction

    @classmethod
    def clone(cls, objective, **kwargs):
        """
        Make a copy of an objective. The objective being copied can be of the same type or belong to
        a different solver interface.

        Example
        ----------
        >>> new_objective = Objective.clone(old_objective)
        """
        return cls(expression=objective.expression, name=objective.name,
                   direction=objective.direction, sloppy=True, **kwargs)

    @property
    def value(self):
        """Return the objective value if it exists."""
        return self._solver.get_objective_value()

    def _direction2str(self):
        if self.direction == "max":
            return "Maximize"
        else:
            return "Minimize"

    def __str__(self):
        return "{}:\n\t{}".format(self._direction2str(), str(self.expression))

    def __repr__(self):
        return "<{} '{}: {}'>".format(
            type(self).__name__, self._direction2str(), str(self.expression))

    # def __eq__(self, other):
    # FIXME: Defining the comparison operation breaks the hash method.
    # FIXME: Do we need the comparison? If so, `__hash__` also needs to be
    # FIXME: defined.
    #     """Tests *mathematical* equality for two Objectives. Solver specific type does NOT have to match.
    #     Expression and direction must be the same.
    #     Name does not have to match"""
    #     if isinstance(other, type(self)):
    #         return self.direction == other.direction and \
    #                self.expression == other.expression
    #     else:
    #         return False

    def _canonicalize(self, expression):
        """For example, changes x + y to 1.*x + 1.*y"""
        expression = super(Objective, self)._canonicalize(expression)
        if isinstance(expression, Basic):
            expression *= 1.
        else:  # pragma: no cover   # symengine
            expression = (1. * expression).expand()
        return expression

    @property
    def direction(self):
        """Return the direction of optimization. Either 'min' or 'max'."""
        return self._direction

    @direction.setter
    def direction(self, value):
        """Set the direction of optimization. Either 'min' or 'max'."""
        if value.lower().startswith("max"):
            self._direction = "max"
        elif value.lower().startswith("min"):
            self._direction = "min"
        else:
            raise ValueError(
                "Provided optimization direction '{}' is neither 'min' nor "
                "'max'.".format(value))

    def set_linear_coefficients(self, coefficients):
        """Set linear coefficients in objective.

        coefficients : dict
            A dictionary of the form {variable1: coefficient1, variable2: coefficient2, ...}
        """
        raise NotImplementedError("Child class should implement this.")

    def to_dict(self):
        """
        Returns a json-compatible object from the objective that can be saved using the json module.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(obj.to_json(), outfile)
        """
        json_obj = {
            "name": self.name,
            "expression": expr_to_json(self.expression),
            "direction": self.direction
        }
        return json_obj

    @classmethod
    def from_dict(cls, json_obj, variables=None):
        """
        Constructs an Objective from the provided json-object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     obj = Objective.from_json(json.load(infile))
        """
        if variables is None:
            variables = {}
        expression = parse_expr(json_obj["expression"], variables)
        return cls(
            expression,
            direction=json_obj["direction"],
            name=json_obj["name"]
        )

