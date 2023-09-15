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

import numpy as np

import optlang.matrix_interface as mi
from optlang import interface
from optlang.exceptions import SolverError


log = logging.getLogger(__name__)

try:
    import highspy as hs
except ImportError:
    raise ImportError("The hybrid interface requires HIGHS and highspy!")

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
    "Unknown": interface.UNDEFINED,
}


_LP_METHODS = ("auto", "simplex", "interior point")


HIGHS_OPTION_MAP = {
    "presolve": {True: "on", False: "off", "auto": "choose"},
    "solver": {"simplex": "simplex", "interior point": "ipm", "auto": "choose"},
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
        d = float(self.direction)
        sp = self.build(add_variable_constraints=True)
        solver = osqp.OSQP()
        solver.setup(
            P=sp.P, q=sp.q, A=sp.A, l=sp.bounds[:, 0], u=sp.bounds[:, 1], **settings
        )
        if self._solution is not None:
            if self.still_valid(sp.A, sp.bounds):
                solver.warm_start(x=self._solution["x"], y=self._solution["y"])
                if "rho" in self._solution:
                    solver.update_settings(rho=self._solution["rho"])
        solution = solver.solve()
        nc = len(self.constraints)
        nv = len(self.variables)
        if not solution.x[0] is None:
            self.primals = dict(zip(self.variables, solution.x))
            self.vduals = dict(zip(self.variables, solution.y[nc : (nc + nv)] * d))
            if nc > 0:
                self.cprimals = dict(zip(self.constraints, sp.A.dot(solution.x)[0:nc]))
                self.duals = dict(zip(self.constraints, solution.y[0:nc] * d))
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
        d = float(self.direction)
        options = self.highs_settings()
        sp = self.build()
        env = self._highs_env
        model = hs.HighsModel()
        env.passOptions(options)
        model.lp_.sense_ = hs.ObjSense.kMinimize
        model.lp_.num_col_ = len(self.variables)
        model.lp_.num_row_ = len(self.constraints)
        # Set variable bounds and objective coefficients
        model.lp_.col_cost_ = sp.q
        model.lp_.col_lower_ = sp.vbounds[:, 0]
        model.lp_.col_upper_ = sp.vbounds[:, 1]
        # Set constraints and bounds
        if sp.A is not None:
            model.lp_.a_matrix_.start_ = sp.A.indptr
            model.lp_.a_matrix_.index_ = sp.A.indices
            model.lp_.a_matrix_.value_ = sp.A.data
            model.lp_.row_lower_ = sp.bounds[:, 0]
            model.lp_.row_upper_ = sp.bounds[:, 1]
        if len(self.integer_vars) > 0:
            model.lp_.integrality_ = HIGHS_VAR_TYPES[sp.integer]
        env.passModel(model)
        env.run()
        info = env.getInfo()
        self.status = env.modelStatusToString(env.getModelStatus())
        sol = env.getSolution()
        primals = np.array(sol.col_value)
        vduals = np.array(sol.col_dual) * d
        cprimals = np.array(sol.row_value)
        duals = np.array(sol.row_dual) * d
        self._solution = {
            "x": primals,
            "y": np.concatenate((duals, vduals)) * d,
        }
        self.primals = dict(zip(self.variables, primals))
        self.vduals = dict(zip(self.variables, vduals))
        self.cprimals = dict(zip(self.constraints, cprimals))
        self.duals = dict(zip(self.constraints, duals))
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

    def __setstate__(self, state):
        state["_highs_env"] = hs.Highs()
        self.__dict__ = state

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["_highs_env"]
        return d


class Variable(mi.Variable):
    pass


class Constraint(mi.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False


class Objective(mi.Objective):
    pass


class Configuration(mi.Configuration):
    lp_methods = _LP_METHODS


class Model(mi.Model):
    ProblemClass = HybridProblem
    status_map = _STATUS_MAP
