# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import pickle
import unittest

from optlang.interface import Variable, Constraint, Model


class VariableTestCase(unittest.TestCase):
    def setUp(self):
        self.var = Variable('x')

    def test_white_space_name_raises(self):
        with self.assertRaises(ValueError):
            Variable('White Space')

    def test_init(self):
        self.assertEqual(self.var.name, 'x')
        self.assertEqual(self.var.lb, None)
        self.assertEqual(self.var.ub, None)
        self.assertEqual(self.var.type, 'continuous')
        self.assertEqual(self.var.problem, None)

        y = Variable('y', lb=-10)
        self.assertEqual(y.lb, -10)
        self.assertEqual(y.ub, None)

        u = Variable('u', ub=-10)
        self.assertEqual(u.lb, None)
        self.assertEqual(u.ub, -10)

    def test_change_lb(self):
        self.var.lb = 666
        self.assertEqual(self.var.lb, 666)
        self.var.lb = None
        self.assertEqual(self.var.lb, None)

    def test_change_ub(self):
        self.var.ub = -465.3323
        self.assertEqual(self.var.ub, -465.3323)
        self.var.ub = None
        self.assertEqual(self.var.ub, None)

    def test_change_type(self):
        self.var.type = 'integer'
        self.assertEqual(self.var.type, 'integer')
        self.var.type = 'binary'
        self.assertEqual(self.var.type, 'binary')
        self.var.type = 'continuous'
        self.assertEqual(self.var.type, 'continuous')
        self.assertRaises(ValueError, setattr, self.var, 'type', 'unicorn')

    def test_change_problem(self):
        model = Model()
        self.var.problem = model
        self.assertEqual(self.var.problem, model)

    def test_lb_greater_than_ub_raises(self):
        self.var.lb = -10
        self.var.ub = 10
        self.assertRaises(Exception, setattr, self.var, 'lb', 20)

    def test_ub_smaller_than_lb_raises(self):
        self.var.lb = 0
        self.var.ub = 10
        self.assertRaises(Exception, setattr, self.var, 'ub', -20)

    def test_pickle_ability(self):
        var = Variable('name', type='binary')
        pickle_var = pickle.loads(pickle.dumps(var))
        keys = var.__dict__.keys()
        self.assertEqual([getattr(var, k) for k in keys], [getattr(pickle_var, k) for k in keys])


# noinspection PyPep8Naming
class ConstraintTestCase(unittest.TestCase):
    def setUp(self):
        self.x = Variable('x', type='binary')
        self.y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        self.z = Variable('z', lb=0., ub=3., type='integer')

    def test_is_Linear(self):
        constraint = Constraint(
            0.3 * self.x + 0.4 * self.y + self.y + 66. * self.z, lb=-100, ub=0., name='linear_constraint')
        self.assertTrue(constraint.is_Linear)
        self.assertFalse(constraint.is_Quadratic)

    def test_is_Quadratic_pow(self):
        constraint = Constraint(
            0.3 * self.x + 0.4 * self.y ** 2 + self.y + 66. * self.z, lb=-100, ub=0., name='quad_pow_constraint')
        self.assertFalse(constraint.is_Linear)
        self.assertTrue(constraint.is_Quadratic)

    def test_is_Quadratic_xy(self):
        constraint = Constraint(
            0.3 * self.x + 0.4 * self.y * self.x + self.y + 66. * self.z, lb=-100, ub=0., name='quad_xy_constraint')
        self.assertFalse(constraint.is_Linear)
        self.assertTrue(constraint.is_Quadratic)
        constraint = Constraint(
            0.3 * self.x ** 2 + 0.4 * self.y + self.y + 66. * self.z, lb=-100, ub=0., name='quad_xy_constraint')
        self.assertFalse(constraint.is_Linear)
        self.assertTrue(constraint.is_Quadratic)

    def test_catching_nonlinear_expressions(self):
        constraint = Constraint(
            0.3 * self.x ** 3 + 0.4 * self.y * self.x + self.y + 66. * self.z, lb=-100, ub=0.,
            name='nonlinear_constraint')
        self.assertFalse(constraint.is_Linear)
        self.assertFalse(constraint.is_Quadratic)
        constraint = Constraint(
            0.3 * self.x ** self.y + 0.4 * self.y * self.x + self.y + 66. * self.z, lb=-100, ub=0.,
            name='nonlinear_constraint')
        self.assertFalse(constraint.is_Linear)
        self.assertFalse(constraint.is_Quadratic)

    def test_lonely_coefficient_and_no_bounds_raises(self):
        self.assertRaises(ValueError, Constraint, self.x + 0.3, name='lonely_coeff_constraint')

    def test_canonicalization_with_lb(self):
        constraint = Constraint(-20 + self.x + 3, lb=-666)
        self.assertEqual(constraint.lb, -649)
        self.assertEqual(constraint.ub, None)
        self.assertEqual(constraint.expression, self.x)

    def test_canonicalization_with_ub(self):
        constraint = Constraint(-20 + self.x + 3, ub=-666)
        self.assertEqual(constraint.ub, -649)
        self.assertEqual(constraint.lb, None)
        self.assertEqual(constraint.expression, self.x)

    def test_canonicalization_with_lb_and_ub(self):
        constraint = Constraint(-20 + self.x + 3, ub=666, lb=-666)
        self.assertEqual(constraint.lb, -649)
        self.assertEqual(constraint.ub, 683)
        self.assertEqual(constraint.expression, self.x)

        # def test_pickle_ability(self):
        # constraint = Constraint(
        #         0.3*self.x**self.y + 0.4*self.y*self.x + self.y + 66.*self.z, lb=-100, ub=0., name='nonlinear_constraint')
        #     pickle_constraint = pickle.loads(pickle.dumps(constraint))
        #     keys = constraint.__dict__.keys()
        #     self.assertEqual([getattr(constraint, k) for k in keys], [getattr(pickle_constraint, k) for k in keys])


if __name__ == '__main__':
    import nose

    nose.runmodule()
