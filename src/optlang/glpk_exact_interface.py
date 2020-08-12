# Copyright 2017 Novo Nordisk Foundation Center for Biosustainability,
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


"""
Interface for the GNU Linear Programming Kit (GLPK)

GLPK is an open source LP solver, with MILP capabilities. This interface exposes its GLPK's exact solver.
To use GLPK you need to install the 'swiglpk' python package (with pip or from http://github.com/biosustain/swiglpk)
and make sure that 'import swiglpk' runs without error.
"""

import logging

import six

from optlang.util import inheritdocstring
from optlang import interface
from optlang import glpk_interface
from optlang.glpk_interface import _GLPK_STATUS_TO_STATUS

log = logging.getLogger(__name__)

from swiglpk import glp_exact, glp_create_prob, glp_get_status, \
    GLP_SF_AUTO, GLP_ETMLIM, glp_adv_basis, glp_read_lp, glp_scale_prob


@six.add_metaclass(inheritdocstring)
class Variable(glpk_interface.Variable):
    def __init__(self, name, index=None, type="continuous", **kwargs):
        if type in ("integer", "binary"):
            raise ValueError("The GLPK exact solver does not support integer and mixed integer problems")
        super(Variable, self).__init__(name, index, type=type, **kwargs)

    @glpk_interface.Variable.type.setter
    def type(self, value):
        if value in ("integer", "binary"):
            raise ValueError("The GLPK exact solver does not support integer and mixed integer problems")
        super(Variable, Variable).type.fset(self, value)


@six.add_metaclass(inheritdocstring)
class Constraint(glpk_interface.Constraint):
    pass


@six.add_metaclass(inheritdocstring)
class Objective(glpk_interface.Objective):
    pass


@six.add_metaclass(inheritdocstring)
class Configuration(glpk_interface.Configuration):
    pass


@six.add_metaclass(inheritdocstring)
class Model(glpk_interface.Model):
    def _run_glp_exact(self):
        return_value = glp_exact(self.problem, self.configuration._smcp)
        glpk_status = glp_get_status(self.problem)
        if return_value == 0:
            status = _GLPK_STATUS_TO_STATUS[glpk_status]
        elif return_value == GLP_ETMLIM:
            status = interface.TIME_LIMIT
        else:
            status = _GLPK_STATUS_TO_STATUS[glpk_status]
            if status == interface.UNDEFINED:
                log.debug("Status undefined. GLPK status code returned by glp_simplex was %d" % return_value)
        return status

    def _optimize(self):
        # Solving inexact first per GLPK manual
        #    Computations in exact arithmetic are very time consuming, so solving LP
        #    problem with the routine glp_exact from the very beginning is not a good
        #    idea. It is much better at first to find an optimal basis with the routine
        #    glp_simplex and only then to call glp_exact, in which case only a few
        #    simplex iterations need to be performed in exact arithmetic.
        status = super(Model, self)._optimize()
        if status != interface.OPTIMAL:
            return status
        else:
            status = self._run_glp_exact()

            if status == interface.UNDEFINED and self.configuration.presolve is True:
                # If presolve is on, status will be undefined if not optimal
                self.configuration.presolve = False
                status = self._run_glp_exact()
                self.configuration.presolve = True
            return status


if __name__ == '__main__':
    import pickle

    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0, ub=1, type='binary')
    c1 = Constraint(x1 + x2 + x3, lb=-100, ub=100, name='c1')
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])
    model.configuration.verbosity = 3
    status = model.optimize()
    print("status:", model.status)
    print("objective value:", model.objective.value)

    for var_name, var in model.variables.items():
        print(var_name, "=", var.primal)

    print(model)

    problem = glp_create_prob()
    glp_read_lp(problem, None, "tests/data/model.lp")

    solver = Model(problem=problem)
    print(solver.optimize())
    print(solver.objective)

    import time

    t1 = time.time()
    print("pickling")
    pickle_string = pickle.dumps(solver)
    resurrected_solver = pickle.loads(pickle_string)
    t2 = time.time()
    print("Execution time: %s" % (t2 - t1))

    resurrected_solver.optimize()
    print("Halelujah!", resurrected_solver.objective.value)
