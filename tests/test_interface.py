# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

from unittest import TestCase
from optlang.interface import Model, Variable, Constraint, Objective


class TestSolver(TestCase):

    def setUp(self):

        self.model = Model()
        x = Variable('x', lb=0, ub=10)
        y = Variable('y', lb=0, ub=10)
        constr = Constraint(x + y, lb=3, name="constr1")
        obj = Objective(2*x + y)
        self.model.add(x)
        self.model.add(y)
        self.model.add(obj)

    def test_create_empty_model(self):
        model = Model()
        self.assertEqual(len(model.constraints), 0)
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(model.objective, None)

    def test_add_variable(self):
        self.assertEqual(self.model.variables['x'].lb, 0)
        self.assertEqual(self.model.variables['x'].ub, 10)
        self.assertEqual(self.model.variables['y'].lb, 0)
        self.assertEqual(self.model.variables['y'].ub, 10)
        var = Variable('x')
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables['x'].problem, var.problem)
        var = Variable('y', lb=-13)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables['x'].lb, None)
        self.assertEqual(self.model.variables['x'].ub, None)
        self.assertEqual(self.model.variables['y'].lb, -13)
        self.assertEqual(self.model.variables['x'].ub, None)

    def test_remove_variable_str(self):
        var = self.model.variables.values()[0]
        self.model.remove(var.name)
        self.assertNotIn(var, self.model.variables.values())
        self.assertEqual(var.problem, None)

    def test_number_objective(self):
        self.model.objective = Objective(0.)
        self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.0')