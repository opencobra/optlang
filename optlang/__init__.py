# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

__version__ = 'v0.0.0'

from .util import list_available_solvers
available_solvers = list_available_solvers()

if available_solvers['GLPK']:
   from .glpk_interface import Model, Variable, Constraint, Objective
elif available_solvers['GUROBI']:
   from .gurobi_interface import Model, Variable, Constraint, Objective
else:
   raise Exception('No solvers available.')

__all__ = []