# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import os
import unittest
import random
import nose

try:
    from optlang.cplex_interface import Variable, Constraint, Model
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

        def test_init_from_existing_problem(self):
            inner_prob = self.model.problem
            self.assertEqual(len(self.model.variables), inner_prob.variables.get_num())
            self.assertEqual(len(self.model.constraints), inner_prob.linear_constraints.get_num() + inner_prob.quadratic_constraints.get_num())
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

        def test_remove_variable(self):
            var = self.model.variables.values()[0]
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.assertNotIn(var, self.model.variables.values())
            self.assertEqual(var.problem, None)

        # def test_add_constraint(self):
        #     x = Variable('x', lb=-83.3, ub=1324422., type='binary')
        #     y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        #     z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        #     constr = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')
        #     self.model.add(constr)

        def test_add_constraints(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constr1 = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')
            # constr1 = Constraint(x + 2* y  + 3.333*z, lb=-10, name='hongo')
            constr2 = Constraint(2.333*x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333*x + y + z, ub=100.33, lb=-300)
            constr4 = constraint = Constraint(0.3*x + 0.4*y**2 + 66.*z, lb=-100, ub=0., name='test_q')
            self.model.add(constr1)
            self.model.add(constr2)
            self.model.add(constr3)
            self.model.add(constr4)
            self.assertIn(constr1, self.model.constraints.values())
            self.assertIn(constr2, self.model.constraints.values())
            self.assertIn(constr3, self.model.constraints.values())
            self.assertIn(constr4, self.model.constraints.values())
            import ipdb; ipdb.set_trace()
            cplex_lines = [line.strip() for line in str(self.model).split('\n')]
            self.assertIn('test: + 0.3 x + 66 z + 0.4 y - ~r_73 = -100', cplex_lines)
            self.assertIn('test2: + 2.333 x + y <= 96.997', cplex_lines)
            self.assertIn('Dummy_14: + 2.333 x + y + z - ~r_75 = -300', cplex_lines)
            


        # def test_add_nonlinear_constraint_raises(self):
        #     x = Variable('x', lb=-83.3, ub=1324422., type='binary')
        #     y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        #     z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        #     constraint = Constraint(0.3*x + 0.4*y**2 + 66.*z, lb=-100, ub=0., name='test')
        #     self.assertRaises(ValueError, self.model.add, constraint)

        # def test_change_variable_bounds(self):
        #     inner_prob = self.model.problem
        #     inner_problem_bounds = [(glp_get_col_lb(inner_prob, i), glp_get_col_ub(inner_prob, i)) for i in range(1, glp_get_num_cols(inner_prob)+1)]
        #     bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
        #     self.assertEqual(bounds, inner_problem_bounds)
        #     for var in self.model.variables.values():
        #         var.lb = random.uniform(-1000, 1000)
        #         var.ub = random.uniform(var.lb, 1000)
        #     inner_problem_bounds_new = [(glp_get_col_lb(inner_prob, i), glp_get_col_ub(inner_prob, i)) for i in range(1, glp_get_num_cols(inner_prob)+1)]
        #     bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
        #     self.assertNotEqual(bounds, bounds_new)
        #     self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
        #     self.assertEqual(bounds_new, inner_problem_bounds_new)

        # def test_change_constraint_bounds(self):
        #     inner_prob = self.model.problem
        #     inner_problem_bounds = [(glp_get_row_lb(inner_prob, i), glp_get_row_ub(inner_prob, i)) for i in range(1, glp_get_num_rows(inner_prob)+1)]
        #     bounds = [(constr.lb, constr.ub) for constr in self.model.constraints.values()]
        #     self.assertEqual(bounds, inner_problem_bounds)
        #     for constr in self.model.constraints.values():
        #         constr.lb = random.uniform(-1000, constr.ub)
        #         constr.ub = random.uniform(constr.lb, 1000)
        #     inner_problem_bounds_new = [(glp_get_row_lb(inner_prob, i), glp_get_row_ub(inner_prob, i)) for i in range(1, glp_get_num_rows(inner_prob)+1)]
        #     bounds_new = [(constr.lb, constr.ub) for constr in self.model.constraints.values()]
        #     self.assertNotEqual(bounds, bounds_new)
        #     self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
        #     self.assertEqual(bounds_new, inner_problem_bounds_new)

        # def test_objective(self):
        #     self.assertEqual(self.model.objective.expression.__str__(), '1.0*R_Biomass_Ecoli_core_w_GAM')

        # def test_optimize(self):
        #     self.model.optimize()
        #     self.assertEqual(self.model.status, 'optimal')
        #     self.assertAlmostEqual(self.model.objective.value, 0.8739215069684303)
except ImportError, e:
    if e.message.find('cplex') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + e.message)
            def test_fail():
                pass
    else:
        raise

if __name__ == '__main__':
    nose.runmodule()
