# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
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

"""Solver interface for the OSQP solver.

Wraps the OSQP solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.

To use this interface, install the OSQP solver and the bundled python
interface.
Make sure that 'import osqp' runs without error.
"""
import logging
import sys

import six
import os
from six.moves import StringIO
from numpy import nan, array, concatenate, Infinity

import osqp

from optlang import interface
from optlang import symbolics
from optlang.util import inheritdocstring, TemporaryFilename
from optlang.expression_parsing import parse_optimization_expression
from optlang.exceptions import SolverError

import pickle
from scipy.sparse import csc_matrix

log = logging.getLogger(__name__)

from optlang.symbolics import add, mul, One, Zero

_STATUS_MAP = {
    'interrupted by user': interface.ABORTED,
    'run time limit reached': interface.TIME_LIMIT,
    'feasible': interface.FEASIBLE,
    'feasible_relaxed_inf': interface.SPECIAL,
    'feasible_relaxed_quad': interface.SPECIAL,
    'feasible_relaxed_sum': interface.SPECIAL,
    'first_order': interface.SPECIAL,
    'primal infeasible': interface.INFEASIBLE,
    'dual infeasible': interface.INFEASIBLE,
    'primal infeasible inaccurate': interface.INFEASIBLE,
    'dual infeasible inaccurate': interface.INFEASIBLE,
    'solved inaccurate': interface.NUMERIC,
    'solved': interface.OPTIMAL,
    'maximum iterations reached': interface.ITERATION_LIMIT,
    'unsolved': interface.SPECIAL,
    'problem non convex': interface.SPECIAL,
    'non-existing-status': 'Here for testing that missing statuses are handled.'
}

_LP_METHODS = ["auto", "qdldl", "mkl pardiso"]

_QP_METHODS = ("auto", )

_TYPES = ["continuous"]


class OSQPProblem(object):
    variables = set()
    constraints = set()
    constraint_coefs = dict()
    constraint_lbs = dict()
    constraint_ubs = dict()
    variable_lbs = dict()
    variable_ubs = dict()
    obj_linear_coefs = dict()
    obj_quadratic_coefs = dict()
    primals = list()
    duals = list()
    obj_value = None
    direction = -1
    info = None
    status = None
    settings = {
        "linsys_solver": "qdldl",
        "max_iter": 10000,
        "eps_abs": 1e-6,
        "eps_rel": 1e-6,
        "eps_prim_inf": 1e-6,
        "eps_dual_inf": 1e-6,
        "polish": True,
        "verbose": False,
        "scaling": 8,
        "time_limit": 0
    }

    def solve(self):
        d = self.direction
        self.clean()
        vmap = {k: i for i, k in enumerate(self.variables)}
        cmap = {k: i for i, k in enumerate(self.constraints)}
        P = array([
            [vmap[vn[0]], vmap[vn[1]], coef * d * (1.0 + vn[0] == vn[1])]
            for vn, coef in self.obj_quadratic_coefs.items()
        ])
        P = csc_matrix((P[:, 2], (P[:, 0], P[:, 1])))
        q = array([
            [vmap[vn], 0, coef * d]
            for vn, coef in self.obj_linear_coefs.items()
        ])
        q = csc_matrix((q[:, 2], (q[:, 0], q[:, 1])))
        q = q.toarray()[:, 0]
        A = array([
            [cmap[vn[0]], vmap[vn[1]], coef * d * 2.0]
            for vn, coef in self.constraint_coefs.items()
        ])
        Av = array([
            [i, i, 1.0] for i in vmap.values()
        ])
        A = concatenate((A, Av), axis=0)
        A = csc_matrix((A[:, 2], (A[:, 0], A[:, 1])))
        lb = array([self.constraint_lbs[k] for k in cmap] +
                   [self.variable_lbs[k] for k in vmap])
        ub = array([self.constraint_ubs[k] for k in cmap] +
                   [self.variable_ubs[k] for k in vmap])
        solution = osqp.solve(
            P=P, q=q, A=A, l=lb, u=ub, **self.settings)  # noqa
        self.primals = solution.x
        self.duals = solution.y
        self.status = solution.info.status
        self.info = solution.info

    def clean(self):
        self.variable_lbs = {k: v for k, v in self.variable_lbs.items()
                             if k in self.variables}
        self.variable_ubs = {k: v for k, v in self.variable_ubs.items()
                             if k in self.variables}
        self.constraint_coefs = {k: v for k, v in self.constraint_coefs.items()
                                 if k[0] in self.constraints and
                                 k[1] in self.variables}
        self.obj_linear_coefs = {k: v for k, v in self.obj_linear_coefs.items()
                                 if k in self.variables}
        self.obj_quadratic_coefs = {
            k: v for k, v in self.obj_quadratic_coefs.items()
            if k[0] in self.variables and k[1] in self.variables}
        self.constraint_lbs = {k: v for k, v in self.constraint_lbs.items()
                               if k[0] in self.constraints}
        self.constraint_ubs = {k: v for k, v in self.constraint_ubs.items()
                               if k[0] in self.constraints}

    def rename_constraint(self, old, new):
        self.constraints.remove(old)
        self.constraints.add(new)
        name_map = {k: k for k in self.constraints}
        name_map[old] = new
        self.constraint_coefs = {
            (name_map[k[0]], k[1]): v
            for k, v in self.constraint_coefs.items()
        }
        self.constraint_lbs[new] = self.constraint_lbs[old]
        self.constraint_ubs[new] = self.constraint_ubs[old]

    def rename_variable(self, old, new):
        self.variables.remove(old)
        self.variables.add(new)
        name_map = {k: k for k in self.variables}
        name_map[old] = new
        self.constraint_coefs = {
            (k[0], name_map[k[1]]): v
            for k, v in self.constraint_coefs.items()
        }
        self.obj_quadratic_coefs = {
            (name_map[k[0]], name_map[k[1]]): v
            for k, v in self.obj_quadratic_coefs.items()
        }
        self.obj_linear_coefs = {
            name_map[k]: v
            for k, v in self.obj_linear_coefs.items()
        }
        self.constraint_lbs[new] = self.constraint_lbs[old]
        self.constraint_ubs[new] = self.constraint_ubs[old]


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @interface.Variable.type.setter
    def type(self, value):
        if self.problem is not None:
            if value not in _TYPES:
                raise ValueError(
                    "OSQP cannot handle variables of type '%s'. " % value +
                    "The following variable types are available: " +
                    ", ".join(_TYPES)
                )
        super(Variable, Variable).type.fset(self, value)

    def _get_primal(self):
        return self.problem.problem.primals[
            self.problem.problem.variables[self.name]]

    @property
    def dual(self):
        if self.problem is None:
            return nan
        return self.problem.problem.duals[
            self.problem.problem.variables[self.name]]

    @interface.Variable.name.setter
    def name(self, value):
        old_name = getattr(self, "name", None)
        super(Variable, Variable).name.fset(self, value)
        if getattr(self, "problem", None) is not None:
            if old_name != value:
                self.problem.problem.rename_variable(old_name, value)


@six.add_metaclass(inheritdocstring)
class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Constraint, self).__init__(
            expression, *args, sloppy=sloppy, **kwargs)

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            triplets = [
                (self.name, var.name, float(coeff))
                for var, coeff in six.iteritems(coefficients)
            ]
            self.problem._set_constraint_coefficient(triplets)
        else:
            raise Exception(
                "Can't change coefficients if constraint is not associated "
                "with a model.")

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = self.problem.problem._get_constraint_coefficients(
                [(self.name, v.name) for v in variables])
            return {v: c for v, c in zip(variables, coefs)}
        else:
            raise Exception(
                "Can't get coefficients from solver if constraint is not "
                "in a model")

    def _get_expression(self):
        if self.problem is not None:
            if self.name not in self.problem.constraints:
                raise ValueError("There is no constraint with name `%s` :(",
                                 self.name)
            variables = self.problem._variables
            coefs = self.problem._get_constraint_coefficients(
                [(self.name, v.name) for v in variables]
            )
            expression = add(
                [mul((symbolics.Real(coefs[i]), v)) for i, v in
                 enumerate(variables)])
            self._expression = expression
        return self._expression

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        if value is None:
            # Update expression from solver instance one last time
            self._get_expression()
        self._problem = value

    @property
    def primal(self):
        if self.problem is None:
            return nan
        var_primals = {v: v._get_primal() for v in self.variables}
        p = self._expression.subs(var_primals).n(16, real=True)
        return p

    @property
    def dual(self):
        raise ValueError("Not supported with OSQP :(")

    @interface.Constraint.name.setter
    def name(self, value):
        old_name = self.name
        super(Constraint, Constraint).name.fset(self, value)
        if getattr(self, "problem", None) is not None:
            if old_name != value:
                self.problem.problem.rename_constraint(old_name, value)

    @interface.Constraint.lb.setter
    def lb(self, value):
        self._check_valid_lower_bound(value)
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.constraint_lbs[
                self.problem.problem.constraints[self.name]] = value
        self._lb = value

    @interface.Constraint.ub.setter
    def ub(self, value):
        self._check_valid_upper_bound(value)
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.constraint_ubs[
                self.problem.problem.constraints[self.name]] = value
        self._ub = value

    def __iadd__(self, other):
        # if self.problem is not None:
        #     self.problem._add_to_constraint(self.index, other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            super(Constraint, self).__iadd__(other)
            problem_reference._add_constraint(self, sloppy=False)
        else:
            super(Constraint, self).__iadd__(other)
        return self


@six.add_metaclass(inheritdocstring)
class Objective(interface.Objective):
    def __init__(self, expression, sloppy=False, **kwargs):
        super(Objective, self).__init__(expression, sloppy=sloppy, **kwargs)
        self._expression_expired = False
        if not (sloppy or self.is_Linear or self.is_Quadratic):
            raise ValueError(
                "OSQP only supports linear and quadratic objectives.")

    @property
    def value(self):
        if getattr(self, 'problem', None) is None:
            return nan
        return self.problem.problem.obj_value

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, Objective).direction.__set__(self, value)
        if value == "min":
            self.problem.problem.direction = 1
        else:
            self.problem.problem.direction = -1

    def _get_expression(self):
        if (self.problem is not None and self._expression_expired and
                len(self.problem._variables) > 0):
            model = self.problem
            vars = {v.name: v for v in model._variables}
            expression = add([
                coef * vars[vn]
                for vn, coef in model.problem.obj_linear_coefficients.items()
                if coef != 0.0
            ])
            if len(model.problem.obj_quadratic_coefs) > 0:
                q_ex = expression = add([
                    coef * vars[vn[0]] * vars[vn[1]]
                    for vn, coef in
                    model.problem.obj_quadratic_coefficients.items()
                    if coef != 0.0
                ])
                expression += q_ex
            self._expression = expression
            self._expression_expired = False
        return self._expression

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            self.problem._set_linear_obj_coefficients(
                [(v.name, float(coef)) for v, coef in coefficients.items()])
            self._expression_expired = True
        else:
            raise Exception(
                "Can't change coefficients if objective is not associated "
                "with a model.")

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = self.problem._get_linear_obj_coefficients(
                [v.name for v in variables])
            return {v: c for v, c in zip(variables, coefs)}
        else:
            raise Exception(
                "Can't get coefficients from solver if objective "
                "is not in a model")


@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, lp_method="qdldl", presolve="auto", verbosity=0,
                 timeout=None, qp_method="auto", *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.lp_method = lp_method
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout
        self.qp_method = qp_method
        if "tolerances" in kwargs:
            for key, val in six.iteritems(kwargs["tolerances"]):
                setattr(self.tolerances, key, val)

    @property
    def lp_method(self):
        """The algorithm used to solve LP problems."""
        lpmethod = self.problem.problem.settings["linsys_solver"]
        return lpmethod

    @lp_method.setter
    def lp_method(self, lp_method):
        if lp_method not in _LP_METHODS:
            raise ValueError(
                "LP Method %s is not valid (choose one of: %s)" %
                (lp_method, ", ".join(_LP_METHODS)))
        self.problem.problem.settings["linsys_solver"] = lp_method

    def _set_presolve(self, value):
        if getattr(self, 'problem', None) is not None:
            if value is True:
                self.problem.problem.settings["scaling"] = 8
            elif value is False or value == "auto":
                self.problem.problem.settings["scaling"] = 0
            else:
                raise ValueError(
                    str(value) +
                    " is not a valid presolve parameter. Must be True, "
                    "False or 'auto'.")

    @property
    def presolve(self):
        return self.problem.problem.settings["scaling"] > 0

    @presolve.setter
    def presolve(self, value):
        self._set_presolve(value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self.problem.problem.settings["verbose"] = value > 0
        self._verbosity = value

    @property
    def timeout(self):
        return self.problem.problem.settings["time_limit"]

    @timeout.setter
    def timeout(self, value):
        if getattr(self, 'problem', None) is not None:
            if value is None:
                self.problem.problem.settings["time_limit"] = 0
            else:
                self.problem.problem.settings["time_limit"] = value

    def __getstate__(self):
        return {"presolve": self.presolve,
                "timeout": self.timeout,
                "verbosity": self.verbosity,
                "tolerances": {"feasibility": self.tolerances.feasibility,
                               "optimality": self.tolerances.optimality,
                               "integrality": self.tolerances.integrality}
                }

    def __setstate__(self, state):
        for key, val in six.iteritems(state):
            if key != "tolerances":
                setattr(self, key, val)

    @property
    def qp_method(self):
        """Change the algorithm used to optimize QP problems."""
        return "auto"

    @qp_method.setter
    def qp_method(self, value):
        if value not in _QP_METHODS:
            raise ValueError(
                "%s is not a valid qp_method. Choose between %s" %
                (value, str(_QP_METHODS)))
        self._qp_method = value

    def _get_feasibility(self):
        return self.problem.problem.settings["eps_prim_inf"]

    def _set_feasibility(self, value):
        self.problem.problem.settings["eps_prim_inf"] = value
        self.problem.problem.settings["eps_dual_inf"] = value

    def _get_integrality(self):
        return None

    def _set_integrality(self, value):
        pass

    def _get_optimality(self):
        return self.problem.problem.settings["eps_abs"]

    def _set_optimality(self, value):
        self.problem.problem.settings["eps_abs"] = value
        self.problem.problem.settings["eps_rel"] = value


    def _tolerance_functions(self):
        return {
            "feasibility": (
                self._get_feasibility,
                self._set_feasibility
            ),
            "optimality": (
                self._get_optimality,
                self._set_optimality
            ),
            "integrality": (
                self._get_integrality,
                self._set_integrality
            )
        }


@six.add_metaclass(inheritdocstring)
class Model(interface.Model):
    def _initialize_problem(self):
        self.problem = OSQPProblem()

    def _initialize_model_from_problem(self, problem):
        if not isinstance(problem, OSQPProblem):
            raise TypeError("Provided problem is not a valid OSQP model.")
        self.problem = problem
        for name in self.problem.variables:
            var = Variable(name, lb=self.problem.variable_lbs[name],
                           ub=self.problem.variable_ubs[name],
                           problem=self)
            super(Model, self)._add_variables([var])

        for name in self.problem.constraints:
            # Since constraint expressions are lazily retrieved from the solver they don't have to be built here
            # lhs = _unevaluated_Add(*[val * variables[i - 1] for i, val in zip(row.ind, row.val)])
            lhs = symbolics.Integer(0)
            constr = Constraint(lhs, lb=self.problem.constraint_lbs[name],
                                ub=self.problem.constraint_ubs[name],
                                name=name, problem=self)
            for variable in constr.variables:
                try:
                    self._variables_to_constraints_mapping[
                        variable.name].add(name)
                except KeyError:
                    self._variables_to_constraints_mapping[
                        variable.name] = set([name])

            super(Model, self)._add_constraints(
                [constr],
                sloppy=True
            )

            linear_expression = add([
                coef * self._variables[vn]
                for vn, coef in self.problem.obj_linear_coefficients.items()
            ])
            quadratic_expression = add([
                coef * self._variables[vn[0]] * self._variables[vn[1]]
                for vn, coef in
                self.problem.obj_quadratic_coefficients.items()
            ])

            self._objective = Objective(
                linear_expression + quadratic_expression,
                problem=self,
                direction=
                {-1: 'max', 1: 'min'}[self.problem.direction],
                name = "osqp_objective"
            )

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        self.update()
        value.problem = None
        if self._objective is not None:  # Reset previous objective
            self.problem.problem.obj_linear_coefs = dict()
            self.problem.problem.obj_quadratic_coefs = dict()
        super(Model, self.__class__).objective.fset(self, value)
        self.update()
        expression = self._objective._expression
        offset, linear_coefficients, quadratic_coeffients = (
            parse_optimization_expression(
                value, quadratic=True, expression=expression)
        )
        self._objective_offset = offset
        if linear_coefficients:
            self.problem.obj_linear_coefs = {
                v.name: c for v, c in linear_coefficients
            }

        for key, coef in quadratic_coeffients.items():
            if len(key) == 1:
                var = six.next(iter(key))
                self.problem.obj_quadratic_coefs[(var.name, var.name)] = \
                    float(coef)
            else:
                var1, var2 = key
                self.problem.obj_quadratic_coefs[(var1.name, var2.name)] = \
                    float(coef)

        self._set_objective_direction(value.direction)
        self.problem.objective.set_name(value.name)
        value.problem = self

    def _set_objective_direction(self, direction):
        self.problem.direction = {'min': 1, 'max': -1}[direction]

    def _get_primal_values(self):
        primal_values = self.problem.primals
        if len(primal_values) == 0:
            raise SolverError("The problem has not been solved yet!")
        return primal_values

    def _get_reduced_costs(self):
        dual_values = self.problem.duals
        if len(dual_values) == 0:
            raise SolverError("The problem has not been solved yet!")
        return dual_values

    def _get_constraint_values(self):
        if len(self.problem.primals) == 0:
            raise SolverError("The problem has not been solved yet!")
        return [c.primal for c in self.constraints]


    def _get_shadow_prices(self):
        if len(self.problem.primals) == 0:
            raise SolverError("The problem has not been solved yet!")
        return [nan for c in self.constraints]

    @property
    def is_integer(self):
        return False

    def _optimize(self):
        self.problem.solve()
        osqp_status = self.problem.status
        self._original_status = osqp_status
        status = _STATUS_MAP[osqp_status]
        return status

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        for var, val in var_lb:
            lb = -Infinity if val is None else val
            self.problem.variable_lbs[var.name] = lb
        for var, val in var_ub:
            ub = Infinity if val is None else val
            self.problem.variable_ubs[var.name] = ub

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        lb, ub = list(), list()
        for variable in variables:
            lb = -Infinity if variable.lb is None else variable.lb
            ub = Infinity if variable.ub is None else variable.ub
            self.problem.variables.add(variable.name)
            self.problem.variable_lbs[variable.name] = lb
            self.problem.variable_ubs[variable.name] = ub
            variable.problem = self

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal from sympy expressions
        if self.objective is not None:
            self.objective._expression = self.objective.expression.xreplace(
                {var: 0 for var in variables})
        for variable in variables:
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]
            self.problem.variables.remove(variable.name)

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)
        for constraint in constraints:
            constraint._problem = None  # This needs to be done in order to not trigger constraint._get_expression()
            if constraint.is_Linear:
                offset, coeff_dict, _ = parse_optimization_expression(constraint)
                self.problem.constraint_coefs.extend({
                    (constraint.name, v.name): co
                    for v, co in coeff_dict.items()
                })
            elif constraint.is_Quadratic:
                raise NotImplementedError(
                    "Quadratic constraints (like %s) are not supported "
                    "in OSQP yet." % constraint)
            else:
                raise ValueError(
                    "OSQP only supports linear or quadratic constraints. "
                    "%s is neither linear nor quadratic." % constraint)
            constraint.problem = self

    def _remove_constraints(self, constraints):
        super(Model, self)._remove_constraints(constraints)
        for constraint in constraints:
            self.problem.constraints.remove(constraint.name)

    def _get_quadratic_expression(self, quadratic=None):
        if quadratic is None:
            vars = self._variables
            return add(co * vars[k[0]] * vars[[1]]
                       for k, co in self.problem.obj_quadratic_coefs)
        terms = []
        vars = list(self.problem.variables)
        for i, sparse_pair in enumerate(quadratic):
            for j, val in zip(sparse_pair.ind, sparse_pair.val):
                i_name, j_name = vars[i], vars[j]
                if i <= j:
                    terms.append(val * self._variables[i_name] *
                                 self._variables[j_name])
                else:
                    pass  # Only look at upper triangle
        return add(terms)

    def _get_variable_indices(self, names):
        vmap = {vn: i for i, vn in enumerate(self.problem.variables)}
        return [vmap[n] for n in names]
