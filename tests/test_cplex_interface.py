# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.
import copy

import unittest
import random
import pickle

import os
import nose
from nose.tools import nottest


try:
    from optlang.cplex_interface import Variable, Constraint, Model, Objective
    import cplex

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')

    class SolverTestCase(unittest.TestCase):

        def setUp(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            assert problem.variables.get_num() > 0
            self.model = Model(problem=problem)

        def test_create_empty_model(self):
            model = Model()
            self.assertEqual(len(model.constraints), 0)
            self.assertEqual(len(model.variables), 0)
            self.assertEqual(model.objective, None)

        def test_copy(self):
            model_copy = copy.copy(self.model)
            self.assertNotEqual(id(self.model), id(model_copy))
            self.assertNotEqual(id(self.model.problem), id(model_copy.problem))

        def test_pickle_ability(self):
            self.model.optimize()
            value = self.model.objective.value
            pickle_string = pickle.dumps(self.model)
            from_pickle = pickle.loads(pickle_string)
            from_pickle.optimize()
            self.assertAlmostEqual(value, from_pickle.objective.value)
            self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in from_pickle.variables.values()],
                             [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
            self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in from_pickle.constraints.values()],
                             [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints.values()])

        def test_init_from_existing_problem(self):
            inner_prob = self.model.problem
            self.assertEqual(len(self.model.variables), inner_prob.variables.get_num())
            self.assertEqual(len(self.model.constraints),
                             inner_prob.linear_constraints.get_num() + inner_prob.quadratic_constraints.get_num())
            self.assertEqual(self.model.variables.keys(), inner_prob.variables.get_names())
            self.assertEqual(self.model.constraints.keys(), inner_prob.linear_constraints.get_names())

        def test_add_variable(self):
            var = Variable('x')
            self.assertEqual(var.problem, None)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(self.model.variables['x'].problem, var.problem)
            self.assertEqual(self.model.variables['x'].problem, self.model)
            var = Variable('y', lb=-13)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(self.model.variables['x'].lb, None)
            self.assertEqual(self.model.variables['x'].ub, None)
            self.assertEqual(self.model.variables['y'].lb, -13)
            self.assertEqual(self.model.variables['x'].ub, None)

        def test_add_integer_var(self):
            var = Variable('int_var', lb=-13, ub=499.4, type='integer')
            self.model.add(var)
            self.assertEqual(self.model.variables['int_var'].type, 'integer')
            self.assertEqual(self.model.variables['int_var'].ub, 499.4)
            self.assertEqual(self.model.variables['int_var'].lb, -13)

        def test_add_non_cplex_conform_variable(self):
            var = Variable('12x!!@#5_3', lb=-666, ub=666)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(var.name, self.model.problem.variables.get_names(len(self.model.variables) - 1))
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            print repickled.variables
            var_from_pickle = repickled.variables['12x!!@#5_3']
            # self.assertEqual(var_from_pickle.name, glp_get_col_name(repickled.problem, var_from_pickle.index))

        def test_remove_variable(self):
            var = self.model.variables.values()[0]
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.assertNotIn(var, self.model.variables.values())
            self.assertEqual(var.problem, None)

        # def test_add_constraint(self):
        # x = Variable('x', lb=-83.3, ub=1324422., type='binary')
        #     y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        #     z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        #     constr = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')
        #     self.model.add(constr)

        def test_add_linear_constraints(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            # constr1 = Constraint(x + 2* y  + 3.333*z, lb=-10, name='hongo')
            constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y + z, ub=100.33, lb=-300)
            self.model.add(constr1)
            self.model.add(constr2)
            self.model.add(constr3)
            self.assertIn(constr1, self.model.constraints.values())
            self.assertIn(constr2, self.model.constraints.values())
            self.assertIn(constr3, self.model.constraints.values())
            cplex_lines = [line.strip() for line in str(self.model).split('\n')]
            self.assertIn('test:       0.4 y + 66 z + 0.3 x - Rgtest  = -100', cplex_lines)
            self.assertIn('test2:      y + 2.333 x <= 96.997', cplex_lines)
            # Dummy_21:   y + z + 2.333 x - RgDummy_21  = -300
            self.assertRegexpMatches(str(self.model), '\s*Dummy_\d+:\s*y \+ z \+ 2\.333 x - .*  = -300')
            print self.model

        @unittest.skip
        def test_add_quadratic_constraints(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constr1 = Constraint(0.3 * x * y + 0.4 * y**2 + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y**2 + z + 33, ub=100.33, lb=-300)
            self.model.add(constr1)
            self.model.add(constr2)
            self.model.add(constr3)
            self.assertIn(constr1, self.model.constraints.values())
            self.assertIn(constr2, self.model.constraints.values())
            self.assertIn(constr3, self.model.constraints.values())
            cplex_lines = [line.strip() for line in str(self.model).split('\n')]
            self.assertIn('test:       0.4 y + 66 z + 0.3 x - Rgtest  = -100', cplex_lines)
            self.assertIn('test2:      y + 2.333 x <= 96.997', cplex_lines)
            # Dummy_21:   y + z + 2.333 x - RgDummy_21  = -300
            self.assertRegexpMatches(str(self.model), '\s*Dummy_\d+:\s*y \+ z \+ 2\.333 x - .*  = -300')
            print self.model

        def test_remove_constraints(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            self.assertEqual(constr1.problem, None)
            self.model.add(constr1)
            self.assertEqual(constr1.problem, self.model)
            self.assertIn(constr1, self.model.constraints.values())
            self.model.remove(constr1.name)
            self.assertEqual(constr1.problem, None)
            self.assertNotIn(constr1, self.model.constraints.values())

        def test_add_nonlinear_constraint_raises(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constraint = Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
            self.assertRaises(ValueError, self.model.add, constraint)

        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x')
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y')]), [0.3, 0.4])
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constraint += 77. * z
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y'), ('test', 'z')]), [0.3, 0.4, 77.])
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')
            print self.model

        def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constraint += 77. * z
            self.model.remove(constraint)
            self.assertEqual(constraint.__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')

        @nottest
        def test_change_of_objective_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            objective = Objective(0.3 * x + 0.4 * y, name='test', direction='max')
            self.model.objective = objective
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x')
            self.assertIn(' obj: + 0.4 y + 0.3 x', self.model.__str__().split("\n"))
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            objective.expression += 77. * z
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x + 77.0*z')
            self.assertIn(' obj: + 0.4 y + 0.3 x + 77. z', self.model.__str__().split("\n"))

except ImportError, e:

    if e.message.find('cplex') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + e.message)
            def test_fail(self):
                pass
    else:
        raise

if __name__ == '__main__':
    nose.runmodule()
