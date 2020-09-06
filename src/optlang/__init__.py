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
import traceback
from optlang._version import get_versions
from optlang.util import list_available_solvers
from optlang.interface import statuses
import optlang.duality

__version__ = get_versions()['version']
del get_versions

log = logging.getLogger(__name__)

# Dictionary of available solvers
available_solvers = list_available_solvers()

# Try to load each solver.
if available_solvers['GLPK']:
    try:
        from optlang import glpk_interface
        from optlang import glpk_exact_interface
    except Exception:
        log.error('GLPK is available but could not load with error:\n  ' + str(traceback.format_exc()).strip().replace('\n','\n  '))

if available_solvers['CPLEX']:
    try:
        from optlang import cplex_interface
    except Exception:
        log.error('CPLEX is available but could not load with error:\n  ' + str(traceback.format_exc()).strip().replace('\n','\n  '))

if available_solvers['GUROBI']:
    try:
        from optlang import gurobi_interface
    except Exception:
        log.error('GUROBI is available but could not load with error:\n  ' + str(traceback.format_exc()).strip().replace('\n','\n  '))

if available_solvers['SCIPY']:
    try:
        from optlang import scipy_interface
    except Exception:
        log.error('SCIPY is available but could not load with error:\n  ' + str(traceback.format_exc()).strip().replace('\n','\n  '))


# Go through and find the best solver that loaded. Load that one as the default
for engine_str in ['cplex_interface', 'gurobi_interface', 'glpk_interface', 'scipy_interface']:
    # Must check globals since not all interface variables will be defined
    if engine_str in globals():
        engine = globals()[engine_str]
        Model = engine.Model
        Variable = engine.Variable
        Constraint = engine.Constraint
        Objective = engine.Objective
        break
else:
    # We were unable to find any valid solvers
    log.error('No solvers were available and/or loadable.')
