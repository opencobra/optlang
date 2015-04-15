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

import six

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import logging
log = logging.getLogger(__name__)

from .util import list_available_solvers

available_solvers = list_available_solvers()

if available_solvers['GLPK']:
    from .glpk_interface import Model, Variable, Constraint, Objective
elif available_solvers['CPLEX']:
    from .cplex_interface import Model, Variable, Constraint, Objective
else:
    log.error('No solvers available.')

if available_solvers['GLPK']:
    if six.PY3:
        from . import glpk_interface
    else:
        import glpk_interface
if available_solvers['CPLEX']:
    if six.PY3:
        from . import cplex_interface
    else:
        import cplex_interface