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

available_solvers = list_available_solvers()

# Load classes from preferred solver interface
if available_solvers['CPLEX']:
    from optlang.cplex_interface import Model, Variable, Constraint, Objective
elif available_solvers["GUROBI"]:
    from optlang.gurobi_interface import Model, Variable, Constraint, Objective
elif available_solvers['GLPK']:
    from optlang.glpk_interface import Model, Variable, Constraint, Objective
elif available_solvers['SCIPY']:
    from optlang.scipy_interface import Model, Variable, Constraint, Objective
else:
    log.error('No solvers available.')

# Import all available solver interfaces
if available_solvers['GLPK']:
    from optlang import glpk_interface
if available_solvers['CPLEX']:
    from optlang import cplex_interface
if available_solvers['GUROBI']:
    from optlang import gurobi_interface
if available_solvers['SCIPY']:
    from optlang import scipy_interface
