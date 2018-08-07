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

from optlang.container import Container
from optlang.interface.variable import Variable
from optlang.interface.constraint import Constraint
from optlang.interface.objective import Objective
from optlang.interface.trackers import (
    VariableChangeTracker, ConstraintChangeTracker, ObjectiveChangeTracker)

__all__ = ("Model",)


class Model(object):
    """
    Represent an optimization problem.

    The model object represents an optimization problem and contains the
    variables, constraints an objective that make up the problem. Variables and
    constraints can be added and removed using the :code:`.add` and
    :code:`.remove` methods, while the objective can be changed by setting the
    objective attribute, e.g.,
    :code:`model.objective = Objective(expr, direction="max")`.

    Once the problem has been formulated the optimization can be performed by
    calling the :code:`.optimize` method. This will return the status of the
    optimization, most commonly 'optimal', 'infeasible' or 'unbounded'.

    Attributes
    ----------
    objective: str
        The objective function.
    name: str, optional
        The name of the optimization problem.
    variables: Container, read-only
        Contains the variables of the optimization problem.
        The keys are the variable names and values are the actual variables.
    constraints: Container, read-only
         Contains the variables of the optimization problem.
         The keys are the constraint names and values are the actual constraints.
    status: str, read-only
        The status of the optimization problem.

    Examples
    --------
    >>> model = Model(name="my_model")
    >>> x1 = Variable("x1", lb=0, ub=20)
    >>> x2 = Variable("x2", lb=0, ub=10)
    >>> c1 = Constraint(2 * x1 - x2, lb=0, ub=0) # Equality constraint
    >>> model.add([x1, x2, c1])
    >>> model.objective = Objective(x1 + x2, direction="max")
    >>> model.optimize()
    'optimal'
    >>> x1.primal, x2.primal
    '(5.0, 10.0)'

    """

    def __init__(self, name=None, objective=None, variables=None,
                 constraints=None, **kwargs):
        super(Model, self).__init__(**kwargs)
        self._variables = Container()
        self._var2constraints = dict()
        self._status = None
        self.name = name
        self._change_set = None
        self._variable_changes = VariableChangeTracker()
        self._constraint_changes = ConstraintChangeTracker()
        self._objective_changes = ObjectiveChangeTracker()
        self._additive_mode = True
        if variables is not None:
            self.add(variables)
        if constraints is not None:
            self.add(constraints)
        if objective is not None:
            self.add(objective)

    def _add_one(self, elem):
        if isinstance(elem, Variable):
            self._variable_changes.add(elem)
            elem.set_observer(self._variable_changes)
            elem.set_observable(self)
        elif isinstance(elem, Constraint):
            self._constraint_changes.add(elem)
            elem.set_observer(self._constraint_changes)
            elem.set_observable(self)
        elif isinstance(elem, Objective):
            self._objective_changes.add(elem)
            elem.set_observer(self._objective_changes)
            elem.set_observable(self)
        else:
            raise TypeError(
                "Can only add variables and constraints not '{}'."
                "".format(type(elem)))

    def add(self, iterable, sloppy=False):
        if not self._additive_mode:
            self.update()
            self._additive_mode = True
        try:
            for elem in iterable:
                self._add_one(elem)
        except TypeError:
            self._add_one(iterable)

    def _remove_one(self, elem):
        if isinstance(elem, Variable):
            self._variable_changes.remove(elem)
            elem.unset_observable()
            elem.unset_observer()
        elif isinstance(elem, Constraint):
            self._constraint_changes.remove(elem)
            elem.unset_observable()
            elem.unset_observer()
        elif isinstance(elem, Objective):
            self._objective_changes.remove(elem)
            elem.unset_observable()
            elem.unset_observer()
        else:
            raise ValueError(
                "Can only remove variables, constraints, and objectives not "
                "'{}'.".format(repr(elem)))

    def remove(self, iterable):
        if self._additive_mode:
            self.update()
            self._additive_mode = False
        try:
            for elem in iterable:
                self._remove_one(elem)
        except TypeError:
            self._remove_one(iterable)

    def update(self):
        self._add_variables()
        self._add_constraints()

        self._update_variable_ubs()
        self._update_constraint_ubs()
        self._update_variable_lbs()
        self._update_constraint_lbs()
        self._update_variable_bounds()
        self._update_constraint_bounds()

        self._remove_variables()
        self._remove_constraints()

    def optimize(self):
        """
        Solve the optimization problem using the relevant solver back-end.
        The status returned by this method tells whether an optimal solution was found,
        if the problem is infeasible etc. Consult optlang.statuses for more elaborate explanations
        of each status.

        The objective value can be accessed from 'model.objective.value', while the solution can be
        retrieved by 'model.primal_values'.

        Returns
        -------
        status: str
            Solution status.
        """
        self.update()
        status = self._optimize()
        if status != OPTIMAL and self.configuration.presolve == "auto":
            self.configuration.presolve = True
            status = self._optimize()
            self.configuration.presolve = "auto"
        self._status = status
        return status

    def _optimize(self):
        raise NotImplementedError(
            "You're using the high level interface to optlang. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    @property
    def variables(self):
        return self._variables

    def _add_variables(self, variables):
        for variable in variables:
            variable.subject = self
            self._variables.append(variable)
            self._var2constraints[variable.name] = set()
