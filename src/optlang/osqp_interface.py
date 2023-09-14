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
import numpy as np

import optlang.matrix_interface as mi
from optlang import interface

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


class OSQPProblem(mi.MatrixProblem):
    """A concise representation of an OSQP problem.

    OSQP assumes that the problem will be pretty much immutable. This is
    a small intermediate layer based on dictionaries that is fast to modify
    but can also be converted to an OSQP problem without too much hassle.
    """

    def osqp_settings(self):
        """Map internal settings to OSQP settings."""
        settings = {
            "linsys_solver": "qdldl",
            "max_iter": 100000,
            "eps_abs": self.settings["optimality_tolerance"],
            "eps_rel": self.settings["optimality_tolerance"],
            "eps_prim_inf": self.settings["primal_inf_tolerance"],
            "eps_dual_inf": self.settings["dual_inf_tolerance"],
            "polish": True,
            "verbose": int(self.settings["verbose"] > 0),
            "scaling": 10 if self.settings["presolve"] else 0,
            "time_limit": self.settings["time_limit"],
            "adaptive_rho": True,
            "rho": 1.0,
            "alpha": 1.6,
        }
        return settings

    def solve(self):
        """Solve the OSQP problem."""
        settings = self.osqp_settings()
        d = float(self.direction)
        sp = self.build(add_variable_constraints=True)
        solver = osqp.OSQP()
        if sp.P is None:
            # see https://github.com/cvxgrp/cvxpy/issues/898
            settings.update({"adaptive_rho": 0, "rho": 1.0, "alpha": 1.0})
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
            self.vduals = dict(zip(self.variables, solution.y[nc : (nc + nv)]))
            if nc > 0:
                self.cprimals = dict(
                    zip(self.constraints, sp.A.dot(solution.x)[0:nc] * d)
                )
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


class Variable(mi.Variable):
    pass


class Constraint(mi.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False


class Objective(mi.Objective):
    pass


class Configuration(mi.Configuration):
    pass


class Model(mi.Model):
    ProblemClass = OSQPProblem
    status_map = _STATUS_MAP
