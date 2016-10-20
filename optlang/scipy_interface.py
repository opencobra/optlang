# Copyright 2015 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function

from collections import OrderedDict
from itertools import islice

import numpy as np
import six
from optlang import interface
from scipy.optimize import linprog

SCIPY_STATUS = {
    0: interface.OPTIMAL,
    1: interface.ABORTED,
    2: interface.INFEASIBLE,
    3: interface.UNBOUNDED
}


class Problem(object):
    """
    Scipy linprog problem object
    """

    def __init__(self):
        self._objective = OrderedDict()
        self._direction = "min"

        self._rows_to_be_added = None
        self._cols_to_be_added = None

        self._A = np.zeros(shape=[0, 0])
        self.upper_bounds = np.zeros([0])
        self.bounds = OrderedDict()

        self._variables = OrderedDict()

        self._constraints = OrderedDict()

        self._status = None
        self._var_primals = None
        self._slacks = None
        self._f = None

    def _get_var_index(self, name):
        return self._variables[name]

    def _get_constraint_index(self, name):
        return self._constraints[name]

    def _add_row_to_A(self, row):
        # Flush any cols_to_be_added
        # if self._cols_to_be_added is not None:
        #    self._flush_cols_to_add()
        # Add the row to queue
        # if self._rows_to_be_added is None:
        #    self._rows_to_be_added = []
        # self._rows_to_be_added.append(row)
        self._A = np.vstack((self._A, row))

    def _flush_rows_to_add(self):
        self._A = np.vstack([self._A] + self._rows_to_be_added)
        self._rows_to_be_added = None

    def _add_col_to_A(self, col):
        # Flush any rows_to_be_added
        # if self._rows_to_be_added is not None:
        #    self._flush_rows_to_add()
        # Add the col to queue
        # if self._cols_to_be_added is None:
        #    self._cols_to_be_added = []
        # self._cols_to_be_added.append(col)
        self._A = np.hstack((self._A, col))

    def _flush_cols_to_add(self):
        self._A = np.hstack([self._A] + self._cols_to_be_added)
        self._cols_to_be_added = None

    def set_variable_bounds(self, name, lower, upper):
        self.bounds[name] = (lower, upper)
        self._reset_solution()

    def add_variable(self, name):
        if name in self._variables:
            raise ValueError(
                "A variable named " + name + " already exists."
            )
        self._variables[name] = len(self._variables)
        self.bounds[name] = (0, None)

        new_col = np.zeros(shape=[len(self._constraints), 1])
        self._add_col_to_A(new_col)
        self._reset_solution()

    def add_constraint(self, name, coefficients={}, ub=0):
        if name in self._constraints:
            raise ValueError(
                "A constraint named " + name + " already exists."
            )
        self._constraints[name] = len(self._constraints)
        self.upper_bounds = np.append(self.upper_bounds, ub)

        new_row = np.array([[coefficients.get(name, 0) for name in self._variables]])
        self._add_row_to_A(new_row)
        self._reset_solution()

    def remove_variable(self, name):
        index = self._get_var_index(name)
        # Remove from matrix
        self._A = np.delete(self.A, index, 1)
        # Remove from bounds
        del self.bounds[name]
        # Remove from var list
        del self._variables[name]
        self._update_variable_indices()
        self._reset_solution()

    def remove_constraint(self, name):
        index = self._get_constraint_index(name)
        # Remove from matrix
        self._A = np.delete(self.A, index, 0)
        # Remove from upper_bounds
        self.upper_bounds = np.delete(self.upper_bounds, index)
        # Remove from constraint list
        del self._constraints[name]
        self._update_constraint_indices()
        self._reset_solution()

    def set_constraint_bound(self, name, value):
        index = self._get_constraint_index(name)
        self.upper_bounds[index] = value
        self._reset_solution()

    def get_var_primal(self, name):
        if self._var_primals is None:
            return None
        else:
            index = self._get_var_index(name)
            return self._var_primals[index]

    def get_var_dual(self, name):
        raise NotImplementedError(
            "Duals are currently not implemented for the Scipy solver. Please use another interface instead.")

    @property
    def A(self):
        assert self._rows_to_be_added is None or self._cols_to_be_added is None
        if self._rows_to_be_added is not None:
            self._flush_rows_to_add()
        if self._cols_to_be_added is not None:
            self._flush_cols_to_add()
        return self._A

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        self._objective = value
        self._reset_solution()

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = value
        self._reset_solution()

    @property
    def status(self):
        if self._status is None:
            raise RuntimeError("Problem has not been optimized yet")
        return SCIPY_STATUS[self._status]

    def get_constraint_slack(self, name):
        if self._slacks is None:
            return None
        else:
            index = self._get_constraint_index(name)
            return self._slacks(index)

    def optimize(self, method="simplex", verbosity=False, **kwargs):
        c = np.array([self.objective.get(name, 0) for name in self._variables])
        if self.direction == "max":
            c *= -1

        bounds = list(six.itervalues(self.bounds))
        solution = linprog(c, self.A, self.upper_bounds, bounds=bounds, method=method,
                           options={"maxiter": 10000, "disp": verbosity}, **kwargs)
        self._solution = solution

        self._var_primals = solution.x
        try:
            self._slacks = solution.slack
        except AttributeError:
            self._slacks = None
        self._status = solution.status
        self._f = solution.fun

    @property
    def objective_value(self):
        if self._f is None:
            raise RuntimeError("Problem has not been optimized yet")
        if self.direction == "max":
            return -self._f
        else:
            return self._f

    def _update_variable_indices(self, start=0):
        i = start
        for name in islice(self._variables, start):
            self._variables[name] = i
            i += 1

    def _update_constraint_indices(self, start=0):
        i = start
        for name in islice(self._constraints, start):
            self._constraints[name] = i
            i += 1

    def _reset_solution(self):
        self._status = None
        self._var_primals = None
        self._slacks = None
        self._f = None


class Variable(interface.Variable):
    def __init__(self, name, lb=None, ub=None, type="continuous", *args, **kwargs):
        if type != "continuous":
            raise ValueError("Scipy solver only works with continuous variables. Please use another interface")
        super(Variable, self).__init__(name, lb, ub, type, *args, **kwargs)

    @interface.Variable.lb.setter
    def lb(self, value):
        interface.Variable.lb.fset(self, value)
        if self.problem:
            self.problem.problem.set_variable_bounds(self.name, self.lb, self.ub)

    @interface.Variable.ub.setter
    def ub(self, value):
        interface.Variable.ub.fset(self, value)
        if self.problem:
            self.problem.problem.set_variable_bounds(self.name, self.lb, self.ub)

    @interface.Variable.type.setter
    def type(self, value):
        if value != "continuous":
            raise ValueError("Scipy solver only works with continuous variables. Please use another interface")

    @property
    def primal(self):
        if getattr(self, "problem", None) is not None:
            primal = self.problem.problem.get_var_primal(self.name)
            return primal
        else:
            return None

    @property
    def dual(self):
        if getattr(self, "problem", None) is not None:
            dual = self.problem.problem.get_var_dual(self.name)
            return dual
        else:
            return None


class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Constraint, self).__init__(expression, sloppy=sloppy, *args, **kwargs)
        if not sloppy:
            if not self.is_Linear:
                raise ValueError(
                    "Scipy only supports linear constraints. %s is not linear." % self)

    @property
    def upper_constraint_name(self):
        return self.name + "_upper"

    @property
    def lower_constraint_name(self):
        return self.name + "_lower"

    @property
    def primal(self):
        if getattr(self, "problem", None) is not None:
            if self.lb is not None:
                primal = self.lb + self.problem.problem.get_constraint_slack(self.lower_constraint_name)
                return primal
            elif self.ub is not None:
                primal = self.ub - self.problem.problem.get_constraint_slack(self.upper_constraint_name)
                return primal
            else:
                return None
        else:
            return None

    @property
    def dual(self):
        if getattr(self, "problem", None) is not None:
            raise NotImplementedError(
                "Duals have not been implemented for the Scipy solver. Please use another interface."
            )
        else:
            return None

    def __setattr__(self, name, value):  # TODO: Change this to properties when PR #11 is merged
        if getattr(self, "problem", None) is not None:
            if name == "lb":
                if self.lb is None and value is not None:
                    negative_coefficient_dict = {name: -coef for name, coef in self.coefficient_dict().items()}
                    self.problem.problem.add_constraint(self.lower_constraint_name, negative_coefficient_dict,
                                                        ub=-value)
                elif value is None and self.lb is not None:
                    self.problem.problem.remove_constraint(self.lower_constraint_name)
                elif value is not None and self.lb is not None:
                    self.problem.problem.set_constraint_bound(self.lower_constraint_name, -value)
            if name == "ub":
                if self.ub is None and value is not None:
                    self.problem.problem.add_constraint(self.upper_constraint_name, self.coefficient_dict(), ub=value)
                elif value is None and self.ub is not None:
                    self.problem.problem.remove_constraint(self.upper_constraint_name)
                elif value is not None and self.ub is not None:
                    self.problem.problem.set_constraint_bound(self.upper_constraint_name, value)

        super(Constraint, self).__setattr__(name, value)

    def coefficient_dict(self):
        if self.expression.is_Add:
            coefficient_dict = {variable.name: coef for variable, coef in
                                self.expression.as_coefficients_dict().items()}
        elif self.expression.is_Atom and self.expression.is_Symbol:
            coefficient_dict = {self.expression.name: 1}
        elif self.expression.is_Mul and len(self.expression.args) <= 2:
            args = self.expression.args
            coefficient_dict = {args[1]: float(args[0])}
        else:
            raise ValueError("Invalid expression")
        return coefficient_dict


class Objective(interface.Objective):
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)
        if not self.is_Linear:
            raise ValueError(
                "Scipy only supports linear objectives. %s is not linear." % self)

    @property
    def value(self):
        if getattr(self, "problem", None) is not None:
            return self.problem.problem.objective_value
        else:
            return None

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, self.__class__).direction.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self.problem.problem.direction = value

    def coefficient_dict(self):
        if self.expression.is_Add:
            coefficient_dict = {variable.name: coef for variable, coef in
                                self.expression.as_coefficients_dict().items()}
        elif self.expression.is_Atom:
            if self.expression.is_Symbol:
                coefficient_dict = {self.expression.name: 1}
            else:
                coefficient_dict = {}
        elif self.expression.is_Mul and len(self.expression.args) <= 2:
            args = self.expression.args
            coefficient_dict = {args[1].name: float(args[0])}
        else:
            raise ValueError("Invalid expression")
        return coefficient_dict


class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, verbosity=0, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self._verbosity = bool(verbosity)

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = value


class Model(interface.Model):
    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        if problem is None:
            self.problem = Problem()
        else:
            if isinstance(problem, Problem):
                self.problem = problem
            else:
                raise TypeError("Problem must be an instance of scipy_interface.Problem")

        self.configuration = Configuration(problem=self)

    def _add_variable(self, variable):
        super(Model, self)._add_variable(variable)
        self.problem.add_variable(variable.name)
        self.problem.set_variable_bounds(variable.name, variable.lb, variable.ub)

    def _remove_variables(self, variables):
        variable_names = [variable.name for variable in variables]
        super(Model, self)._remove_variables(variables)
        for name in variable_names:
            self.problem.remove_variable(name)

    def _add_constraint(self, constraint, sloppy=False):
        if not (sloppy or constraint.is_Linear):
            raise ValueError("Scipy solver only works with linear constraints. Please use another interface.")

        super(Model, self)._add_constraint(constraint, sloppy=sloppy)
        coefficient_dict = constraint.coefficient_dict()

        if constraint.ub is not None:
            self.problem.add_constraint(constraint.upper_constraint_name, coefficient_dict)
            self.problem.set_constraint_bound(constraint.upper_constraint_name, constraint.ub)

        if constraint.lb is not None:
            negative_coefficient_dict = {name: -coef for name, coef in coefficient_dict.items()}
            self.problem.add_constraint(constraint.lower_constraint_name, negative_coefficient_dict)
            self.problem.set_constraint_bound(constraint.lower_constraint_name, -constraint.lb)

    def _remove_constraints(self, constraints):
        for constraint in constraints:
            if constraint.lb is not None:
                self.problem.remove_constraint(constraint.lower_constraint_name)
            if constraint.ub is not None:
                self.problem.remove_constraint(constraint.upper_constraint_name)

            super(Model, self)._remove_constraints(constraints)

    def optimize(self):
        self.problem.optimize(verbosity=self.configuration.verbosity)
        status = self.problem.status
        self._status = status
        return status

    @interface.Model.objective.setter
    def objective(self, value):
        interface.Model.objective.fset(self, value)
        if value is None:
            self.problem.objective = {}
        else:
            self.problem.objective = self._objective.coefficient_dict()
            self.problem.direction = value.direction


if __name__ == "__main__":
    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)
    c1 = Constraint(x1 + x2 + x3, lb=-100, ub=100, name='c1')
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])
    status = model.optimize()
    print("status:", model.status)
    print("objective value:", model.objective.value)

    for var in model.variables:
        print(var.name, "=", var.primal)
