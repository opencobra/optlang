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

"""Hybrid solver interface combining HIGHS and OSQP for a large-scale LP/QP/MIP solver.

This uses the template from the :mod:`matrix_interface` to have a lean internal
representation that allows the crossover between the solvers..

To use this interface, install the OSQP and HIGHS solvers and the bundled python
interface.
Make sure that `import osqp` and `import highspy` runs without error.
"""
import logging
import six
import numpy as np

import optlang.matrix_interface as mi
from optlang import interface
from optlang.exceptions import SolverError
from optlang.util import inheritdocstring

log = logging.getLogger(__name__)

try:
    import osqp as osqp
    import highspy as hs
except ImportError:
    raise ImportError("The hybrid interface requires highs and osqp!")


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
    "Optimal": interface.OPTIMAL,
    "Not Set": interface.ABORTED,
    "Load error": interface.SPECIAL,
    "Model error": interface.SPECIAL,
    "Presolve error": interface.SPECIAL,
    "Solve error": interface.SPECIAL,
    "Postsolve error": interface.SPECIAL,
    "Empty": interface.UNDEFINED,
    "Infeasible": interface.INFEASIBLE,
    "Primal infeasible or unbounded": interface.INFEASIBLE_OR_UNBOUNDED,
    "Unbounded": interface.UNBOUNDED,
    "Bound on objective reached": interface.OPTIMAL,
    "Target for objective reached": interface.SUBOPTIMAL,
    "Time limit reached": interface.TIME_LIMIT,
    "Iteration limit reached": interface.ITERATION_LIMIT,
    "Solution limit reached": interface.NODE_LIMIT,
    "Unknown": interface.UNDEFINED
}


_LP_METHODS = ("auto", "simplex", "interior point")


HIGHS_OPTION_MAP = {
    "presolve": {True: "on", False: "off", "auto": "choose"},
    "solver": {"simplex": "simplex", "interior point": "ipm", "auto": "choose"}
}

HIGHS_VAR_TYPES = np.array([hs.HighsVarType.kContinuous, hs.HighsVarType.kInteger])


class HybridProblem(mi.MatrixProblem):
    """A concise representation of an OSQP problem.

    OSQP assumes that the problem will be pretty much immutable. This is
    a small intermediate layer based on dictionaries that is fast to modify
    but can also be converted to an OSQP problem without too much hassle.
    """

    def __init__(self):
        super().__init__()
        self._highs_env = hs.Highs()

    def osqp_settings(self):
        """Map internal settings to OSQP settings."""
        settings = {
            "linsys_solver": "qdldl",
            "max_iter": self.settings["iteration_limit"],
            "eps_abs": self.settings["optimality_tolerance"],
            "eps_rel": self.settings["optimality_tolerance"],
            "eps_prim_inf": self.settings["primal_inf_tolerance"],
            "eps_dual_inf": self.settings["dual_inf_tolerance"],
            "polish": True,
            "verbose": int(self.settings["verbose"] > 0),
            "scaling": 10 if self.settings["presolve"] is True else 0,
            "time_limit": self.settings["time_limit"],
            "adaptive_rho": True,
            "rho": 1.0,
            "alpha": 1.6,
        }
        return settings

    def highs_settings(self):
        """Map internal settings to OSQP settings."""
        options = hs.HighsOptions()
        options.primal_feasibility_tolerance = self.settings["primal_inf_tolerance"]
        options.dual_feasibility_tolerance = self.settings["dual_inf_tolerance"]
        options.ipm_optimality_tolerance = self.settings["optimality_tolerance"]
        options.mip_feasibility_tolerance = self.settings["mip_tolerance"]
        options.presolve = HIGHS_OPTION_MAP["presolve"][self.settings["presolve"]]
        options.solver = HIGHS_OPTION_MAP["solver"][self.settings["lp_method"]]
        if self.settings["time_limit"] == 0:
            options.time_limit = float("inf")
        else:
            options.time_limit = float(self.settings["time_limit"])
        options.log_to_console = self.settings["verbose"] > 0
        options.output_flag = self.settings["verbose"] > 0
        options.threads = self.settings["threads"]
        options.ipm_iteration_limit = self.settings["iteration_limit"]
        options.simplex_iteration_limit = self.settings["iteration_limit"] * 1000

        return options

    def solve_osqp(self):
        """Solve a QP with OSQP."""
        settings = self.osqp_settings()
        P, q, A, bounds, _, _ = self.build(add_variable_constraints=True)
        solver = osqp.OSQP()
        if P is None:
            # see https://github.com/cvxgrp/cvxpy/issues/898
            settings.update({"adaptive_rho": 0, "rho": 1.0, "alpha": 1.0})
        solver.setup(P=P, q=q, A=A, l=bounds[:, 0], u=bounds[:, 1], **settings)  # noqa
        if self._solution is not None:
            if self.still_valid(A, bounds):
                solver.warm_start(x=self._solution["x"], y=self._solution["y"])
                if "rho" in self._solution:
                    solver.update_settings(rho=self._solution["rho"])
        solution = solver.solve()
        nc = len(self.constraints)
        nv = len(self.variables)
        if not solution.x[0] is None:
            self.primals = dict(zip(self.variables, solution.x))
            self.vduals = dict(zip(self.variables, solution.y[nc : (nc + nv)]))
            if nc > 0:
                self.cprimals = dict(zip(self.constraints, A.dot(solution.x)[0:nc]))
                self.duals = dict(zip(self.constraints, solution.y[0:nc]))
        if not np.isnan(solution.info.obj_val):
            self.obj_value = solution.info.obj_val * self.direction
            self.status = solution.info.status
        else:
            self.status = "primal infeasible"
            self.obj_value = None
        self.info = solution.info
        self._solution = {
            "x": solution.x,
            "y": solution.y,
            "rho": solution.info.rho_estimate,
        }

    def solve_highs(self):
        """Solve a problem with HIGHS."""
        options = self.highs_settings()
        P, q, A, bounds, vbounds, ints = self.build()
        env = self._highs_env
        model = hs.HighsModel()
        env.passOptions(options)
        model.lp_.num_col_ = len(self.variables)
        model.lp_.num_row_ = len(self.constraints)
        # Set variable bounds and objective coefficients
        model.lp_.col_cost_ = q
        model.lp_.col_lower_ = vbounds[:, 0]
        model.lp_.col_upper_ = vbounds[:, 1]
        # Set constraints and bounds
        if A is not None:
            model.lp_.a_matrix_.start_ = A.indptr
            model.lp_.a_matrix_.index_ = A.indices
            model.lp_.a_matrix_.value_ = A.data
            model.lp_.row_lower_ = bounds[:, 0]
            model.lp_.row_upper_ = bounds[:, 1]
        if len(self.integer_vars) > 0:
            model.lp_.integrality_ = HIGHS_VAR_TYPES[ints]
        env.passModel(model)
        env.run()
        info = env.getInfo()
        self.status = env.modelStatusToString(env.getModelStatus())
        sol = env.getSolution()
        self._solution = {
            "x": list(sol.col_value),
            "y": list(sol.row_dual) + list(sol.col_dual)
        }
        self.primals = dict(zip(self.variables, list(sol.col_value)))
        self.vduals = dict(zip(self.variables, list(sol.col_dual)))
        self.cprimals = dict(zip(self.constraints, list(sol.row_value)))
        self.duals = dict(zip(self.constraints, list(sol.row_dual)))
        self.obj_value = info.objective_function_value * self.direction
        self.info = info

    def solve(self):
        if len(self.obj_quadratic_coefs) == 0:  # linear problem
            self.solve_highs()
        else:
            if len(self.integer_vars) > 0:
                raise SolverError("MIQPs are not supported by the hybrid solver!")
            else:
                self.solve_osqp()

    def still_valid(self, A, bounds):
        """Check if previous solutions is still feasible."""
        if len(self._solution["x"]) != len(self.variables) or len(
            self._solution["y"]
        ) != len(self.constraints):
            return False
        c = A.dot(self._solution["x"])
        ea = self.settings["eps_abs"]
        er = self.settings["eps_rel"]
        valid = np.all(
            (c + er * np.abs(c) + ea >= bounds[:, 0])
            & (c - er * np.abs(c) - ea <= bounds[:, 1])
        )
        return valid

    def reset(self, everything=False):
        super().reset(everything)
        self._highs_env = hs.Highs()


@six.add_metaclass(inheritdocstring)
class Variable(mi.Variable):
    pass


@six.add_metaclass(inheritdocstring)
class Constraint(mi.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False


@six.add_metaclass(inheritdocstring)
class Objective(mi.Objective):
    pass


@six.add_metaclass(inheritdocstring)
class Configuration(mi.Configuration):
    lp_methods = _LP_METHODS


@six.add_metaclass(inheritdocstring)
class Model(mi.Model):
    ProblemClass = HybridProblem
    status_map = _STATUS_MAP
