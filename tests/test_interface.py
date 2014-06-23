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
        obj = Objective(2 * x + y)
        self.model.add(x)
        self.model.add(y)
        self.model.add(obj)

    def test_read_only_attributes(self):
        self.assertRaises(AttributeError, setattr, self.model, 'variables', 'Foo')
        self.assertRaises(AttributeError, setattr, self.model, 'constraints', 'Foo')
        self.assertRaises(AttributeError, setattr, self.model, 'status', 'Foo')

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

    def test_add_differing_interface_type_raises(self):
        from optlang import glpk_interface as glpk
        x, y = glpk.Variable('x'), glpk.Variable('y')
        constraint = glpk.Constraint(x+y)
        objective = glpk.Objective(1.*x)
        self.assertRaises(TypeError, self.model.add, x)
        self.assertRaises(TypeError, self.model.add, constraint)
        self.assertRaises(TypeError, self.model.add, objective)

    def test_setting_binary_bounds(self):
        self.model.variables['x'].type = 'binary'
        self.assertEqual(self.model.variables['x'].lb, 0)
        self.assertEqual(self.model.variables['x'].ub, 1)

    def test_non_integer_bound_on_integer_variable_raises(self):
        self.model.variables['x'].type = 'integer'
        self.assertRaises(ValueError, setattr, self.model.variables['x'], 'lb', 3.3)

    def test_non_0_or_1_bound_on_binary_variable_raises(self):
        self.model.variables['x'].type = 'integer'
        self.assertRaises(ValueError, setattr, self.model.variables['x'], 'lb', 3.3)