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

from __future__ import absolute_import

import logging
from optlang._version import get_versions
from optlang.util import list_available_solvers
from optlang.interface import statuses
import optlang.duality

__version__ = get_versions()['version']
del get_versions

log = logging.getLogger(__name__)

# The try/fail approach doesn't require this, but it is still a useful list
available_solvers = list_available_solvers()

# Try to load each solver.
try:
	from optlang import glpk_interface
except:
	glpk_interface = None
try:
	from optlang import cplex_interface
except:
	cplex_interface = None
try:
	from optlang import gurobi_interface
except:
	gurobi_interface = None
try:
	from optlang import scipy_interface
except:
	scipy_interface = None

# Go through and find the best solver that loaded. Load that one as the default
best_interface = None
for engine in [cplex_interface,gurobi_interface,glpk_interface,scipy_interface]:
	if engine is not None:
		best_interface = engine
		Model = engine.Model
		Variable = engine.Variable
		Constraint = engine.Constraint
		Objective = engine.Objective
		break

# If we can't find any interface, that is probably an issue
if best_interface is None:
	log.error('No solvers were available and/or loadable.')
