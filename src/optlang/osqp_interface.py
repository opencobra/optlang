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
import six
import pickle
import numpy as np
from numpy import array, concatenate, Infinity, zeros, isnan

from optlang import interface, symbolics, available_solvers
from optlang.util import inheritdocstring
from optlang.expression_parsing import parse_optimization_expression
from optlang.exceptions import SolverError

from scipy.sparse import csc_matrix

from optlang.symbolics import add, mul

log = logging.getLogger(__name__)

try:
    import osqp as osqp
except ImportError:
    try:
        import cuosqp as osqp

        log.warning(
            "cuOSQP is still experimental and may not work for all problems. "
            "It may not converge as fast as the normal osqp package or not "
            "converge at all."
        )
    except ImportError:
        raise ImportError("The osqp_interface requires osqp or cuosqp!")


_STATUS_MAP = {
    "interrupted by user": interface.ABORTED,
    "run time limit reached": interface.TIME_LIMIT,
    "feasible": interface.FEASIBLE,
    "primal infeasible": interface.INFEASIBLE,
    "dual infeasible": interface.INFEASIBLE,
    "primal infeasible inaccurate": interface.INFEASIBLE,
    "dual infeasible inaccurate": interface.INFEASIBLE,
    "solved inaccurate": interface.NUMERIC,
    "solved": interface.OPTIMAL,
    "maximum iterations reached": interface.ITERATION_LIMIT,
    "unsolved": interface.SPECIAL,
    "problem non convex": interface.SPECIAL,
    "non-existing-status": "Here for testing that missing statuses are handled.",
}

_LP_METHODS = ("auto", "primal")

_QP_METHODS = ("auto", "primal")

_TYPES = ("continuous",)


class OSQPProblem(object):
    """A concise representation of an OSQP problem.

    OSQP assumes that the problem will be pretty much immutable. This is
    a small intermediate layer based on dictionaries that is fast to modify
    but can also be converted to an OSQP problem without too much hassle.
    """

    def __init__(self):
        self.variables = set()
        self.constraints = set()
        self.constraint_coefs = dict()
        self.constraint_lbs = dict()
        self.constraint_ubs = dict()
        self.variable_lbs = dict()
        self.variable_ubs = dict()
        self.obj_linear_coefs = dict()
        self.obj_quadratic_coefs = dict()
        self.primals = {}
        self.cprimals = {}
        self.duals = {}
        self.vduals = {}
        self.obj_value = None
        self.direction = -1
        self.info = None
        self.status = None
        self.settings = {
            "linsys_solver": "qdldl",
            "max_iter": 100000,
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "eps_prim_inf": 1e-6,
            "eps_dual_inf": 1e-6,
            "polish": True,
            "verbose": False,
            "scaling": 0,
            "time_limit": 0,
            "adaptive_rho": True,
            "rho": 1.0,
            "alpha": 1.6,
        }
        self.__solution = None

    def build(self):
        """Build the problem instance."""
        d = self.direction
        vmap = dict(zip(self.variables, range(len(self.variables))))
        nv = len(self.variables)
        cmap = dict(zip(self.constraints, range(len(self.constraints))))
        nc = len(self.constraints)
        if len(self.obj_quadratic_coefs) > 0:
            P = array(
                [
                    [vmap[vn[0]], vmap[vn[1]], coef * d * 2.0]
                    for vn, coef in six.iteritems(self.obj_quadratic_coefs)
                ]
            )
            P = csc_matrix(
                (P[:, 2], (P[:, 0].astype("int64"), P[:, 1].astype("int64"))),
                shape=(nv, nv),
            )
        else:
            P = None
        q = zeros(nv)
        q[[vmap[vn] for vn in self.obj_linear_coefs]] = list(
            self.obj_linear_coefs.values()
        )
        q = q * d
        Av = array([[vmap[k] + nc, vmap[k], 1.0] for k in self.variables])
        vbounds = array(
            [[self.variable_lbs[vn], self.variable_ubs[vn]] for vn in self.variables]
        )
        if len(self.constraint_coefs) > 0:
            A = array(
                [
                    [cmap[vn[0]], vmap[vn[1]], coef]
                    for vn, coef in six.iteritems(self.constraint_coefs)
                ]
            )
            bounds = array(
                [
                    [self.constraint_lbs[cn], self.constraint_ubs[cn]]
                    for cn in self.constraints
                ]
            )
            A = concatenate((A, Av))
            bounds = concatenate((bounds, vbounds))
        else:
            A = Av
            bounds = vbounds
        if A.shape[0] == 0:
            A = None
        else:
            A = csc_matrix(
                (A[:, 2], (A[:, 0].astype("int64"), A[:, 1].astype("int64"))),
                shape=(nc + nv, nv),
            )
        return P, q, A, bounds

    def solve(self):
        """Solve the OSQP problem."""
        settings = self.settings.copy()
        P, q, A, bounds = self.build()
        solver = osqp.OSQP()
        if P is None:
            # see https://github.com/cvxgrp/cvxpy/issues/898
            settings.update({"adaptive_rho": 0, "rho": 1.0, "alpha": 1.0})
        solver.setup(P=P, q=q, A=A, l=bounds[:, 0], u=bounds[:, 1], **settings)  # noqa
        if self.__solution is not None:
            if self.still_valid(A, bounds):
                solver.warm_start(x=self.__solution["x"], y=self.__solution["y"])
                solver.update_settings(rho=self.__solution["rho"])
        solution = solver.solve()
        nc = len(self.constraints)
        nv = len(self.variables)
        if not solution.x[0] is None:
            self.primals = dict(zip(self.variables, solution.x))
            self.vduals = dict(zip(self.variables, solution.y[nc : (nc + nv)]))
            if nc > 0:
                self.cprimals = dict(zip(self.constraints, A.dot(solution.x)[0:nc]))
                self.duals = dict(zip(self.constraints, solution.y[0:nc]))
        if not isnan(solution.info.obj_val):
            self.obj_value = solution.info.obj_val * self.direction
            self.status = solution.info.status
        else:
            self.status = "primal infeasible"
            self.obj_value = None
        self.info = solution.info
        self.__solution = {
            "x": solution.x,
            "y": solution.y,
            "rho": solution.info.rho_estimate,
        }

    def reset(self, everything=False):
        """Reset the public solver solution."""
        self.info = None
        self.primals = {}
        self.cprimals = {}
        self.duals = {}
        self.vduals = {}
        if everything:
            self.__solution = None

    def still_valid(self, A, bounds):
        """Check if previous solutions is still feasible."""
        if len(self.__solution["x"]) != len(self.variables) or len(
            self.__solution["y"]
        ) != len(self.constraints):
            return False
        c = A.dot(self.__solution["x"])
        ea = self.settings["eps_abs"]
        er = self.settings["eps_rel"]
        valid = np.all(
            (c + er * np.abs(c) + ea >= bounds[:, 0])
            & (c - er * np.abs(c) - ea <= bounds[:, 1])
        )
        return valid

    def clean(self):
        """Remove unused variables and constraints."""
        self.reset()
        self.variable_lbs = {
            k: v for k, v in six.iteritems(self.variable_lbs) if k in self.variables
        }
        self.variable_ubs = {
            k: v for k, v in six.iteritems(self.variable_ubs) if k in self.variables
        }
        self.constraint_coefs = {
            k: v
            for k, v in six.iteritems(self.constraint_coefs)
            if k[0] in self.constraints and k[1] in self.variables
        }
        self.obj_linear_coefs = {
            k: v for k, v in six.iteritems(self.obj_linear_coefs) if k in self.variables
        }
        self.obj_quadratic_coefs = {
            k: v
            for k, v in six.iteritems(self.obj_quadratic_coefs)
            if k[0] in self.variables and k[1] in self.variables
        }
        self.constraint_lbs = {
            k: v for k, v in six.iteritems(self.constraint_lbs) if k in self.constraints
        }
        self.constraint_ubs = {
            k: v for k, v in six.iteritems(self.constraint_ubs) if k in self.constraints
        }

    def rename_constraint(self, old, new):
        """Rename a constraint."""
        self.reset()
        self.constraints.remove(old)
        self.constraints.add(new)
        name_map = {k: k for k in self.constraints}
        name_map[old] = new
        self.constraint_coefs = {
            (name_map[k[0]], k[1]): v for k, v in six.iteritems(self.constraint_coefs)
        }
        self.constraint_lbs[new] = self.constraint_lbs.pop(old)
        self.constraint_ubs[new] = self.constraint_ubs.pop(old)

    def rename_variable(self, old, new):
        """Rename a variable."""
        self.reset()
        self.variables.remove(old)
        self.variables.add(new)
        name_map = {k: k for k in self.variables}
        name_map[old] = new
        self.constraint_coefs = {
            (k[0], name_map[k[1]]): v for k, v in six.iteritems(self.constraint_coefs)
        }
        self.obj_quadratic_coefs = {
            (name_map[k[0]], name_map[k[1]]): v
            for k, v in six.iteritems(self.obj_quadratic_coefs)
        }
        self.obj_linear_coefs = {
            name_map[k]: v for k, v in six.iteritems(self.obj_linear_coefs)
        }
        self.variable_lbs[new] = self.variable_lbs.pop(old)
        self.variable_ubs[new] = self.variable_ubs.pop(old)


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @interface.Variable.type.setter
    def type(self, value):
        if self.problem is not None:
            if value not in _TYPES:
                raise ValueError(
                    "OSQP cannot handle variables of type '%s'. " % value
                    + "The following variable types are available: "
                    + ", ".join(_TYPES)
                )
        super(Variable, Variable).type.fset(self, value)

    def _get_primal(self):
        return self.problem.problem.primals.get(self.name, None)

    @property
    def dual(self):
        if self.problem is not None:
            return self.problem.problem.vduals.get(self.name, None)
        return None

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
        super(Constraint, self).__init__(expression, *args, sloppy=sloppy, **kwargs)

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            self.problem.problem.reset()
            for var, coef in six.iteritems(coefficients):
                self.problem.problem.constraint_coefs[(self.name, var.name)] = float(
                    coef
                )
        else:
            raise Exception(
                "Can't change coefficients if constraint is not associated "
                "with a model."
            )

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = {
                v: self.problem.problem.constraint_coefs.get((self.name, v.name), 0.0)
                for v in variables
            }
            return coefs
        else:
            raise Exception(
                "Can't get coefficients from solver if constraint is not " "in a model"
            )

    def _get_expression(self):
        if self.problem is not None:
            variables = self.problem._variables
            all_coefs = self.problem.problem.constraint_coefs
            coefs = [(v, all_coefs.get((self.name, v.name), 0.0)) for v in variables]
            expression = add([mul((symbolics.Real(co), v)) for (v, co) in coefs])
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
            return None
        return self.problem.problem.cprimals.get(self.name, None)

    @property
    def dual(self):
        if self.problem is None:
            return None
        d = self.problem.problem.duals.get(self.name, None)
        if d is not None:
            d = -d
        return d

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
        if getattr(self, "problem", None) is not None:
            lb = -Infinity if value is None else float(value)
            self.problem.problem.constraint_lbs[self.name] = lb
        self._lb = value

    @interface.Constraint.ub.setter
    def ub(self, value):
        self._check_valid_upper_bound(value)
        if getattr(self, "problem", None) is not None:
            ub = Infinity if value is None else float(value)
            self.problem.problem.constraint_ubs[self.name] = ub
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
            raise ValueError("OSQP only supports linear and quadratic objectives.")

    @property
    def value(self):
        if getattr(self, "problem", None) is None:
            return None
        if self.problem.problem.obj_value is None:
            return None
        return self.problem.problem.obj_value + self.problem._objective_offset

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, Objective).direction.__set__(self, value)
        if value == "min":
            self.problem.problem.direction = 1
        else:
            self.problem.problem.direction = -1

    def _get_expression(self):
        if (
            self.problem is not None
            and self._expression_expired
            and len(self.problem._variables) > 0
        ):
            model = self.problem
            vars = model._variables
            expression = add(
                [
                    coef * vars[vn]
                    for vn, coef in six.iteritems(model.problem.obj_linear_coefs)
                ]
            )
            q_ex = add(
                [
                    coef * vars[vn[0]] * vars[vn[1]]
                    for vn, coef in six.iteritems(model.problem.obj_quadratic_coefs)
                ]
            )
            expression += q_ex
            self._expression = expression
            self._expression_expired = False
        return self._expression

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            for v, coef in six.iteritems(coefficients):
                self.problem.problem.obj_linear_coefs[v.name] = float(coef)
            self._expression_expired = True
        else:
            raise Exception(
                "Can't change coefficients if objective is not associated "
                "with a model."
            )

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = {
                v: self.problem.problem.obj_linear_coefs.get(v.name, 0.0)
                for v in variables
            }
            return coefs
        else:
            raise Exception(
                "Can't get coefficients from solver if objective " "is not in a model"
            )


@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(
        self,
        lp_method="primal",
        presolve=False,
        verbosity=0,
        timeout=None,
        qp_method="primal",
        linear_solver="qdldl",
        *args,
        **kwargs
    ):
        super(Configuration, self).__init__(*args, **kwargs)
        self.lp_method = lp_method
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout
        self.qp_method = qp_method
        self.linear_solver = linear_solver
        if "tolerances" in kwargs:
            for key, val in six.iteritems(kwargs["tolerances"]):
                if key in self._tolerance_functions():
                    setattr(self.tolerances, key, val)

    @property
    def lp_method(self):
        """The algorithm used to solve LP problems."""
        return "primal"

    @lp_method.setter
    def lp_method(self, lp_method):
        if lp_method not in _LP_METHODS:
            raise ValueError(
                "LP Method %s is not valid (choose one of: %s)"
                % (lp_method, ", ".join(_LP_METHODS))
            )

    @property
    def linear_solver(self):
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, solver):
        if solver not in ("qdldl", "mkl pardiso"):
            raise ValueError(
                "%s is not valid (choose either `qdldl` or `mkl pardiso`)" % solver
            )
        if getattr(self, "problem", None) is not None:
            self.problem.problem.settings["linsys_solver"] = solver
        self._linear_solver = solver

    def _set_presolve(self, value):
        if getattr(self, "problem", None) is not None:
            if value is True:
                self.problem.problem.settings["scaling"] = 10
            elif value is False or value == "auto":
                self.problem.problem.settings["scaling"] = 0
            else:
                raise ValueError(
                    str(value) + " is not a valid presolve parameter. Must be True, "
                    "False or 'auto'."
                )

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
        if getattr(self, "problem", None) is not None:
            self.problem.problem.settings["verbose"] = int(value > 0)
        self._verbosity = value

    @property
    def timeout(self):
        return self.problem.problem.settings["time_limit"]

    @timeout.setter
    def timeout(self, value):
        if getattr(self, "problem", None) is not None:
            if value is None:
                self.problem.problem.settings["time_limit"] = 0
            else:
                self.problem.problem.settings["time_limit"] = value

    def __getstate__(self):
        return {
            "presolve": self.presolve,
            "timeout": self.timeout,
            "verbosity": self.verbosity,
            "linear_solver": self.linear_solver,
            "tolerances": {
                "feasibility": self.tolerances.feasibility,
                "optimality": self.tolerances.optimality,
                "integrality": self.tolerances.integrality,
            },
        }

    def __setstate__(self, state):
        for key, val in six.iteritems(state):
            if key != "tolerances":
                setattr(self, key, val)
        for key, val in six.iteritems(state["tolerances"]):
            if key in self._tolerance_functions():
                setattr(self.tolerances, key, val)

    @property
    def qp_method(self):
        """Change the algorithm used to optimize QP problems."""
        return "primal"

    @qp_method.setter
    def qp_method(self, value):
        if value not in _QP_METHODS:
            raise ValueError(
                "%s is not a valid qp_method. Choose between %s"
                % (value, str(_QP_METHODS))
            )
        self._qp_method = value

    def _get_feasibility(self):
        return self.problem.problem.settings["eps_prim_inf"]

    def _set_feasibility(self, value):
        self.problem.problem.settings["eps_prim_inf"] = value
        self.problem.problem.settings["eps_dual_inf"] = value

    def _get_integrality(self):
        return 1e-6

    def _set_integrality(self, value):
        pass

    def _get_optimality(self):
        return self.problem.problem.settings["eps_abs"]

    def _set_optimality(self, value):
        self.problem.problem.settings["eps_abs"] = value
        self.problem.problem.settings["eps_rel"] = value

    def _tolerance_functions(self):
        return {
            "feasibility": (self._get_feasibility, self._set_feasibility),
            "optimality": (self._get_optimality, self._set_optimality),
            "integrality": (self._get_integrality, self._set_integrality),
        }


@six.add_metaclass(inheritdocstring)
class Model(interface.Model):
    def _initialize_problem(self):
        self.problem = OSQPProblem()

    def _initialize_model_from_problem(self, problem, vc_mapping=None, offset=0):
        if not isinstance(problem, OSQPProblem):
            raise TypeError("Provided problem is not a valid OSQP model.")
        self.problem = problem
        for name in self.problem.variables:
            var = Variable(
                name,
                lb=self.problem.variable_lbs[name],
                ub=self.problem.variable_ubs[name],
                problem=self,
            )
            super(Model, self)._add_variables([var])

        for name in self.problem.constraints:
            # Since constraint expressions are lazily retrieved from the
            # solver they don't have to be built here
            lhs = symbolics.Integer(0)
            constr = Constraint(
                lhs,
                lb=self.problem.constraint_lbs[name],
                ub=self.problem.constraint_ubs[name],
                name=name,
                problem=self,
            )

            super(Model, self)._add_constraints([constr], sloppy=True)

        if vc_mapping is None:
            for constr in self.constraints:
                name = constr.name
                for variable in constr.variables:
                    try:
                        self._variables_to_constraints_mapping[variable.name].add(name)
                    except KeyError:
                        self._variables_to_constraints_mapping[variable.name] = set(
                            [name]
                        )
        else:
            self._variables_to_constraints_mapping = vc_mapping

        linear_expression = add(
            [
                coef * self._variables[vn]
                for vn, coef in six.iteritems(self.problem.obj_linear_coefs)
            ]
        )
        quadratic_expression = add(
            [
                coef * self._variables[vn[0]] * self._variables[vn[1]]
                for vn, coef in six.iteritems(self.problem.obj_quadratic_coefs)
            ]
        )

        self._objective_offset = offset
        self._objective = Objective(
            linear_expression + quadratic_expression + offset,
            problem=self,
            direction={-1: "max", 1: "min"}[self.problem.direction],
            name="osqp_objective",
        )

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        value.problem = None
        if self._objective is not None:  # Reset previous objective
            self.problem.obj_linear_coefs = dict()
            self.problem.obj_quadratic_coefs = dict()
        super(Model, self.__class__).objective.fset(self, value)
        self.update()
        expression = self._objective._expression
        (
            offset,
            linear_coefficients,
            quadratic_coeffients,
        ) = parse_optimization_expression(value, quadratic=True, expression=expression)
        self._objective_offset = offset
        if linear_coefficients:
            self.problem.obj_linear_coefs = {
                v.name: float(c) for v, c in six.iteritems(linear_coefficients)
            }

        for key, coef in six.iteritems(quadratic_coeffients):
            if len(key) == 1:
                var = six.next(iter(key))
                self.problem.obj_quadratic_coefs[(var.name, var.name)] = float(coef)
            else:
                var1, var2 = key
                self.problem.obj_quadratic_coefs[(var1.name, var2.name)] = 0.5 * float(
                    coef
                )
                self.problem.obj_quadratic_coefs[(var2.name, var1.name)] = 0.5 * float(
                    coef
                )

        self._set_objective_direction(value.direction)
        value.problem = self

    def _set_objective_direction(self, direction):
        self.problem.direction = {"min": 1, "max": -1}[direction]

    def _get_primal_values(self):
        if len(self.problem.primals) == 0:
            raise SolverError("The problem has not been solved yet!")
        primal_values = [self.problem.primals[v.name] for v in self._variables]
        return primal_values

    def _get_reduced_costs(self):
        if len(self.problem.duals) == 0:
            raise SolverError("The problem has not been solved yet!")
        reduced_costs = [self.problem.vduals[v.name] for v in self._variables]
        return reduced_costs

    def _get_constraint_values(self):
        if len(self.problem.primals) == 0:
            raise SolverError("The problem has not been solved yet!")
        constraint_primals = [self.problem.cprimals[c.name] for c in self._constraints]
        return constraint_primals

    def _get_shadow_prices(self):
        if len(self.problem.primals) == 0:
            raise SolverError("The problem has not been solved yet!")
        dual_values = [-self.problem.duals[c.name] for c in self._constraints]
        return dual_values

    @property
    def is_integer(self):
        return False

    def _optimize(self):
        self.update()
        self.problem.solve()
        osqp_status = self.problem.status
        self._original_status = osqp_status
        status = _STATUS_MAP[osqp_status]
        return status

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        self.problem.reset()
        for var, val in var_lb:
            lb = -Infinity if val is None else float(val)
            self.problem.variable_lbs[var.name] = lb
        for var, val in var_ub:
            ub = Infinity if val is None else val
            self.problem.variable_ubs[var.name] = float(ub)

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        self.problem.reset()
        for variable in variables:
            lb = -Infinity if variable.lb is None else float(variable.lb)
            ub = Infinity if variable.ub is None else float(variable.ub)
            self.problem.variables.add(variable.name)
            self.problem.variable_lbs[variable.name] = lb
            self.problem.variable_ubs[variable.name] = ub
            variable.problem = self

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal from sympy expressions
        if self.objective is not None:
            self.objective._expression = self.objective.expression.xreplace(
                {var: 0 for var in variables}
            )
        for variable in variables:
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]
            self.problem.variables.remove(variable.name)
        self.problem.clean()

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)
        self.problem.reset()
        for constraint in constraints:
            constraint._problem = None  # This needs to be done in order to not trigger constraint._get_expression()
            if constraint.is_Linear:
                _, coeff_dict, _ = parse_optimization_expression(constraint)
                lb = -Infinity if constraint.lb is None else float(constraint.lb)
                ub = Infinity if constraint.ub is None else float(constraint.ub)
                self.problem.constraints.add(constraint.name)
                self.problem.constraint_coefs.update(
                    {
                        (constraint.name, v.name): float(co)
                        for v, co in six.iteritems(coeff_dict)
                    }
                )
                self.problem.constraint_lbs[constraint.name] = lb
                self.problem.constraint_ubs[constraint.name] = ub
                constraint.problem = self
            elif constraint.is_Quadratic:
                raise NotImplementedError(
                    "Quadratic constraints (like %s) are not supported "
                    "in OSQP yet." % constraint
                )
            else:
                raise ValueError(
                    "OSQP only supports linear or quadratic constraints. "
                    "%s is neither linear nor quadratic." % constraint
                )

    def _remove_constraints(self, constraints):
        super(Model, self)._remove_constraints(constraints)
        for constraint in constraints:
            self.problem.constraints.remove(constraint.name)
        self.problem.clean()

    def _get_variable_indices(self, names):
        vmap = dict(zip(self.variables, range(len(self.variables))))
        return [vmap[n] for n in names]

    def __setstate__(self, d):
        self.__init__()
        osqp = pickle.loads(d["osqp"])
        # workaround to conserve the order
        osqp.variables = d["variables"]
        osqp.constraints = d["constraints"]
        self._initialize_model_from_problem(
            osqp, vc_mapping=d["v_to_c"], offset=d["offset"]
        )
        osqp.variables = set(osqp.variables)
        osqp.constraints = set(osqp.constraints)
        self.configuration = Configuration()
        self.configuration.problem = self
        self.configuration.__setstate__(d["config"])

    def __getstate__(self):
        self.problem.reset()
        self.update()
        return {
            "osqp": pickle.dumps(self.problem),
            "variables": [v.name for v in self._variables],
            "constraints": [c.name for c in self._constraints],
            "v_to_c": self._variables_to_constraints_mapping,
            "config": self.configuration.__getstate__(),
            "offset": getattr(self, "_objective_offset", 0.0),
        }

    @classmethod
    def from_lp(self, lp_problem_str):
        """Read a model from an LP file.

        OSQP does not have an integrated LP reader so it will either use
        cplex or glpk to read the model. This means that QP problems will
        currently require cplex to be read :(
        """
        if available_solvers["CPLEX"]:
            from optlang import cplex_interface

            mod = cplex_interface.Model.from_lp(lp_problem_str)
            mod.configuration.lp_method = "auto"
            mod.configuration.qp_method = "auto"
            return super(Model, self).clone(mod)
        else:
            from optlang import glpk_interface

            mod = glpk_interface.Model.from_lp(lp_problem_str)
            mod.configuration.lp_method = "auto"
            return super(Model, self).clone(mod)
