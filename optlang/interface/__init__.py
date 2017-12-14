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

"""Provide the general interface to optimization problems."""

from __future__ import absolute_import

# from optlang.interface.config import MathematicalProgrammingConfiguration
from optlang.interface.variable import Variable
# from optlang.interface.expression import OptimizationExpression
# from optlang.interface.constraint import Constraint
# from optlang.interface.objective import Objective
# from optlang.interface.model import Model

__all__ = (
#     "MathematicalProgrammingConfiguration",
    "Variable",
#     "OptimizationExpression",
#     "Constraint",
#     "Objective",
#     "Model",
#     "statuses"
)

# # OPTIMAL = 'optimal'
# UNDEFINED = 'undefined'
# FEASIBLE = 'feasible'
# INFEASIBLE = 'infeasible'
# NOFEASIBLE = 'nofeasible'
# UNBOUNDED = 'unbounded'
# INFEASIBLE_OR_UNBOUNDED = 'infeasible_or_unbounded'
# LOADED = 'loaded'
# CUTOFF = 'cutoff'
# ITERATION_LIMIT = 'iteration_limit'
# MEMORY_LIMIT = 'memory_limit'
# NODE_LIMIT = 'node_limit'
# TIME_LIMIT = 'time_limit'
# SOLUTION_LIMIT = 'solution_limit'
# INTERRUPTED = 'interrupted'
# NUMERIC = 'numeric'
# SUBOPTIMAL = 'suboptimal'
# INPROGRESS = 'in_progress'
# ABORTED = 'aborted'
# SPECIAL = 'check_original_solver_status'

# statuses = {
#     OPTIMAL: "An optimal solution as been found.",
#     INFEASIBLE: "The problem has no feasible solutions.",
#     UNBOUNDED: "The objective can be optimized infinitely.",
#     SPECIAL: "The status returned by the solver could not be interpreted. Please refer to the solver's documentation to find the status.",
#     UNDEFINED: "The solver determined that the problem is ill-formed. "
#     # TODO Add the rest
# }
