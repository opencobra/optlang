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


import os
import unittest

import nose
import optlang
import optlang.glpk_interface as glpk
from optlang import interface
from optlang.util import glpk_read_cplex

TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')

try:
    import optlang.cplex_interface as cplex


    class ChangeSolverTestCase(unittest.TestCase):
        def setUp(self):
            self.model = optlang.glpk_interface.Model(problem=glpk_read_cplex(TESTMODELPATH))

        def test_clone_constraint_interface_to_glpk(self):
            x1 = interface.Variable('x1', lb=0)
            x2 = interface.Variable('x2', lb=0)
            x3 = interface.Variable('x3', lb=0)
            c1 = interface.Constraint(x1 + x2 + x3, ub=100)
            glpk_c1 = glpk.Constraint.clone(c1)
            print(glpk_c1)
            self.assertIs(glpk_c1.__class__, glpk.Constraint)
            for variable in glpk_c1.variables:
                self.assertIs(variable.__class__, glpk.Variable)

        def test_clone_constraint_glpk_to_cplex(self):
            x1 = glpk.Variable('x1', lb=0)
            x2 = glpk.Variable('x2', lb=0)
            x3 = glpk.Variable('x3', lb=0)
            c1 = glpk.Constraint(x1 + x2 + x3, ub=100)
            cplex_c1 = cplex.Constraint.clone(c1)
            print(cplex_c1)
            self.assertIs(cplex_c1.__class__, cplex.Constraint)
            for variable in cplex_c1.variables:
                self.assertIs(variable.__class__, cplex.Variable)

        def test_clone_objective_glpk_to_cplex(self):
            x1 = glpk.Variable('x1', lb=0)
            x2 = glpk.Variable('x2', lb=0)
            x3 = glpk.Variable('x3', lb=0)
            obj = glpk.Objective(x1 + x2 + x3)
            cplex_obj = cplex.Objective.clone(obj)
            self.assertIs(cplex_obj.__class__, cplex.Objective)
            for variable in cplex_obj.variables:
                self.assertIs(variable.__class__, cplex.Variable)

        def test_clone_to_cplex(self):
            cplex_model = cplex.Model.clone(self.model)
            self.assertEqual(cplex_model.__class__, cplex.Model)
            for variable in cplex_model.variables.values():
                self.assertIs(variable.__class__, cplex.Variable)
            for constraint in cplex_model.constraints:
                self.assertIs(constraint.__class__, cplex.Constraint)

        def test_clone_to_glpk(self):
            glpk_model = glpk.Model.clone(self.model)
            self.assertEqual(glpk_model.__class__, glpk.Model)
            for variable in glpk_model.variables.values():
                self.assertIs(variable.__class__, glpk.Variable)
            for constraint in glpk_model.constraints:
                self.assertIs(constraint.__class__, glpk.Constraint)

except ImportError as e:

    if str(e).find('cplex') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise

if __name__ == '__main__':
    nose.runmodule()
