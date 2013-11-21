'''

@author: Nikolaus sonnenschein

   Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
   Technical University of Denmark.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
'''

from unittest import TestCase
from optlang.interface import Variable, Constraint, Objective, Solver

# class TestVariable(TestCase):
   
#    def test_set_lb(self):
#       self.var.lb = -1000.
#       self.assertEqual()

# class TestObjective(TestCase):

#    def test 

class TestSolver(TestCase):

    def setUp(self):

        self.solver = Solver()
        x = Variable('x', lb=0, ub=10)
        y = Variable('y', lb=0, ub=10)
        constr = Constraint(x + y > 3, name="constr1")
        obj = Objective(2*x + y)
        self.solver.add(x)
        self.solver.add(y)
        self.solver.add(obj)

    def test_1(self):
        self.assertEqual(1+1, 2)

    def test_solve(self):
        sol = solver.optimize()
        self.assertEqual(sol, 3.)