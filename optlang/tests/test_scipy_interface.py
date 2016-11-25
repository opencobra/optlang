# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import unittest
import json
import os
import optlang.interface

try:
    import scipy
except ImportError as e:
    if str(e).find('scipy') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise
else:
    from optlang import scipy_interface
    from optlang.tests import abstract_test_cases

    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/coli_core.json')


    class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
        interface = scipy_interface

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            with open(TESTMODELPATH) as infile:
                model = self.interface.Model.from_json(json.load(infile))

            model.optimize()
            self.assertEqual(model.status, optlang.interface.OPTIMAL)
            for var in model.variables:
                self.assertTrue(var.lb <= round(var.primal, 6) <= var.ub, (var.lb, var.primal, var.ub))

        def test_changing_variable_names_is_reflected_in_the_solver(self):
            self.skipTest("NA")

        def test_set_wrong_type_raises(self):
            self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
            self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
            self.model.add(self.var)
            self.model.update()
            self.assertRaises(ValueError, setattr, self.var, "type", "mustard")
            # self.var.type = "integer"
            # self.assertEqual(self.var.type, "integer")

        @unittest.skip("Scipy doesn't support duals")
        def test_get_dual(self):
            pass

        def test_change_type(self):
            self.var.type = "continuous"
            self.assertRaises(ValueError, setattr, self.var, "type", "integer")
            self.assertRaises(ValueError, setattr, self.var, "type", "binary")


    class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
        interface = scipy_interface

        def test_get_primal(self):

            with open(TESTMODELPATH) as infile:
                self.model = self.interface.Model.from_json(json.load(infile))
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            for c in self.model.constraints:
                if c.lb is not None:
                    self.assertTrue(c.lb <= round(c.primal, 6))
                if c.ub is not None:
                    self.assertTrue(round(c.primal, 6) <= c.ub)

        @unittest.skip("NA")
        def test_get_dual(self):
            pass

        @unittest.skip("NA")
        def test_indicator_constraint_support(self):
            pass

    class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
        interface = scipy_interface

        def setUp(self):
            with open(TESTMODELPATH) as infile:
                self.model = self.interface.Model.from_json(json.load(infile))
            self.obj = self.model.objective

        def test_change_direction(self):
            self.obj.direction = "min"
            self.assertEqual(self.obj.direction, "min")
            self.assertEqual(self.model.problem.direction, "min")

            self.obj.direction = "max"
            self.assertEqual(self.obj.direction, "max")
            self.assertEqual(self.model.problem.direction, "max")


    class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
        interface = scipy_interface

        def setUp(self):
            with open(TESTMODELPATH) as infile:
                self.model = self.interface.Model.from_json(json.load(infile))

        @unittest.skip("NA")
        def test_clone_model_with_lp(self):
            pass

        @unittest.skip("NA")
        def test_optimize_milp(self):
            pass

        @unittest.skip("Not implemented yet")
        def test_pickle_empty_model(self):
            pass

        @unittest.skip("Not implemented yet")
        def test_pickle_ability(self):
            self.model.optimize()
            value = self.model.objective.value
            pickle_string = pickle.dumps(self.model)
            from_pickle = pickle.loads(pickle_string)
            from_pickle.optimize()
            self.assertAlmostEqual(value, from_pickle.objective.value)
            self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in from_pickle.variables.values()],
                             [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
            self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in from_pickle.constraints],
                             [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

        @unittest.skip("Not implemented yet")
        def test_config_gets_copied_too(self):
            self.assertEquals(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            model_copy = copy.copy(self.model)
            self.assertEquals(model_copy.configuration.verbosity, 3)

        @unittest.skip("Not implemented yet")
        def test_init_from_existing_problem(self):
            inner_prob = self.model.problem
            self.assertEqual(len(self.model.variables), glp_get_num_cols(inner_prob))
            self.assertEqual(len(self.model.constraints), glp_get_num_rows(inner_prob))
            self.assertEqual(self.model.variables.keys(),
                             [glp_get_col_name(inner_prob, i) for i in range(1, glp_get_num_cols(inner_prob) + 1)])
            self.assertEqual(self.model.constraints.keys(),
                             [glp_get_row_name(inner_prob, j) for j in range(1, glp_get_num_rows(inner_prob) + 1)])

        @unittest.skip("Not implemented yet")
        def test_add_non_cplex_conform_variable(self):
            var = Variable('12x!!@#5_3', lb=-666, ub=666)
            self.assertEqual(var.index, None)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(var.name, glp_get_col_name(self.model.problem, var.index))
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            var_from_pickle = repickled.variables['12x!!@#5_3']
            self.assertEqual(var_from_pickle.name, glp_get_col_name(repickled.problem, var_from_pickle.index))

        def test_add_constraints(self):
            x = self.interface.Variable('x', lb=0, ub=1, type='continuous')
            y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = self.interface.Variable('z', lb=0., ub=10., type='continuous')
            constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            constr2 = self.interface.Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
            constr3 = self.interface.Constraint(2.333 * x + y + z, lb=-300)
            constr4 = self.interface.Constraint(x, lb=-300, ub=-300)
            constr5 = self.interface.Constraint(3 * x)
            self.model.add(constr1)
            self.model.add(constr2)
            self.model.add(constr3)
            self.model.add([constr4, constr5])
            self.assertIn(constr1.name, self.model.constraints)
            self.assertIn(constr2.name, self.model.constraints)
            self.assertIn(constr3.name, self.model.constraints)
            self.assertIn(constr4.name, self.model.constraints)
            self.assertIn(constr5.name, self.model.constraints)

        @unittest.skip("")
        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.assertEqual(constraint.index, None)
            self.model.add(constraint)
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x')
            self.assertEqual(constraint.index, 73)
            z = Variable('z', lb=3, ub=10, type='integer')
            self.assertEqual(z.index, None)
            constraint += 77. * z
            self.assertEqual(z.index, 98)
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')
            print(self.model)
            self.assertEqual(constraint.index, 73)

        @unittest.skip("")
        def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            z = Variable('z', lb=2, ub=5, type='integer')
            constraint += 77. * z
            self.model.remove(constraint)
            self.assertEqual(constraint.__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')

        @unittest.skip("")
        def test_change_of_objective_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            objective = Objective(0.3 * x + 0.4 * y, name='test', direction='max')
            self.model.objective = objective
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x')
            self.assertEqual(glp_get_obj_coef(self.model.problem, x.index), 0.3)
            self.assertEqual(glp_get_obj_coef(self.model.problem, y.index), 0.4)
            for i in range(1, glp_get_num_cols(self.model.problem) + 1):
                if i != x.index and i != y.index:
                    self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)
            z = Variable('z', lb=4, ub=4, type='integer')
            self.model.objective += 77. * z
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x + 77.0*z')
            self.assertEqual(glp_get_obj_coef(self.model.problem, x.index), 0.3)
            self.assertEqual(glp_get_obj_coef(self.model.problem, y.index), 0.4)
            self.assertEqual(glp_get_obj_coef(self.model.problem, z.index), 77.)
            for i in range(1, glp_get_num_cols(self.model.problem) + 1):
                if i != x.index and i != y.index and i != z.index:
                    self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)

        @unittest.skip("")
        def test_change_variable_bounds(self):
            inner_prob = self.model.problem
            inner_problem_bounds = [(glp_get_col_lb(inner_prob, i), glp_get_col_ub(inner_prob, i)) for i in
                                    range(1, glp_get_num_cols(inner_prob) + 1)]
            bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertEqual(bounds, inner_problem_bounds)
            for var in self.model.variables.values():
                var.lb = random.uniform(-1000, 1000)
                var.ub = random.uniform(var.lb, 1000)
            inner_problem_bounds_new = [(glp_get_col_lb(inner_prob, i), glp_get_col_ub(inner_prob, i)) for i in
                                        range(1, glp_get_num_cols(inner_prob) + 1)]
            bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        @unittest.skip("")
        def test_change_constraint_bounds(self):
            inner_prob = self.model.problem
            inner_problem_bounds = [(glp_get_row_lb(inner_prob, i), glp_get_row_ub(inner_prob, i)) for i in
                                    range(1, glp_get_num_rows(inner_prob) + 1)]
            bounds = [(constr.lb, constr.ub) for constr in self.model.constraints]
            self.assertEqual(bounds, inner_problem_bounds)
            for constr in self.model.constraints:
                constr.lb = random.uniform(-1000, constr.ub)
                constr.ub = random.uniform(constr.lb, 1000)
            inner_problem_bounds_new = [(glp_get_row_lb(inner_prob, i), glp_get_row_ub(inner_prob, i)) for i in
                                        range(1, glp_get_num_rows(inner_prob) + 1)]
            bounds_new = [(constr.lb, constr.ub) for constr in self.model.constraints]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        def test_initial_objective(self):
            self.assertIn('1.0*BIOMASS_Ecoli_core_w_GAM', self.model.objective.expression.__str__(), )

        @unittest.skip("")
        def test_iadd_objective(self):
            v2, v3 = self.model.variables.values()[1:3]
            self.model.objective += 2. * v2 - 3. * v3
            obj_coeff = list()
            for i in range(len(self.model.variables)):
                obj_coeff.append(glp_get_obj_coef(self.model.problem, i))
            self.assertEqual(obj_coeff,
                             [0.0, 1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0]
                             )

        @unittest.skip("")
        def test_imul_objective(self):
            self.model.objective *= 2.
            obj_coeff = list()
            for i in range(len(self.model.variables)):
                obj_coeff.append(glp_get_obj_coef(self.model.problem, i))
            self.assertEqual(obj_coeff,
                             [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0,
                              0.0]
                             )

        @unittest.skip("Not implemented yet")
        def test_set_copied_objective(self):
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_Biomass_Ecoli_core_w_GAM')

        @unittest.skip("NA")
        def test_timeout(self):
            self.model.configuration.timeout = 0
            status = self.model.optimize()
            self.assertEqual(status, 'time_limit')

        @unittest.skip("")
        def test_set_linear_coefficients_objective(self):
            self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
            self.assertEqual(glp_get_obj_coef(self.model.problem, self.model.variables.R_TPI.index), 666.)

        @unittest.skip("")
        def test_instantiating_model_with_different_solver_problem_raises(self):
            self.assertRaises(TypeError, Model, problem='Chicken soup')

        @unittest.skip("")
        def test_set_linear_coefficients_constraint(self):
            constraint = self.model.constraints.M_atp_c
            constraint.set_linear_coefficients({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
            num_cols = glp_get_num_cols(self.model.problem)
            ia = intArray(num_cols + 1)
            da = doubleArray(num_cols + 1)
            index = constraint.index
            num = glp_get_mat_row(self.model.problem, index, ia, da)
            for i in range(1, num + 1):
                col_name = glp_get_col_name(self.model.problem, ia[i])
                if col_name == 'R_Biomass_Ecoli_core_w_GAM':
                    self.assertEqual(da[i], 666.)

        def test_shadow_prices(self):
            self.model.optimize()
            self.assertRaises(NotImplementedError, getattr, self.model, "shadow_prices")

        def test_reduced_costs(self):
            self.model.optimize()
            self.assertRaises(NotImplementedError, getattr, self.model, "reduced_costs")

        def test_remove_constraints(self):
            x = self.interface.Variable('x', type='continuous')
            y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = self.interface.Variable('z', lb=4, ub=4, type='continuous')
            constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            self.assertEqual(constr1.problem, None)
            self.model.add(constr1)
            self.model.update()
            self.assertEqual(constr1.problem, self.model)
            self.assertIn(constr1, self.model.constraints)
            self.model.remove(constr1.name)
            self.model.update()
            self.assertEqual(constr1.problem, None)
            self.assertNotIn(constr1, self.model.constraints)

        def test_add_nonlinear_constraint_raises(self):
            x = self.interface.Variable('x', type='continuous')
            y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = self.interface.Variable('z', lb=3, ub=3, type='continuous')
            with self.assertRaises(ValueError):
                constraint = self.interface.Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
                self.model.add(constraint)
                self.model.update()

        def test_change_objective(self):
            v1, v2 = self.model.variables.values()[0:2]
            self.model.objective = self.interface.Objective(1. * v1 + 1. * v2)
            self.assertIn(v1.name, str(self.model.objective))
            self.assertIn(v2.name, str(self.model.objective))

        def test_change_variable_type(self):
            self.assertRaises(ValueError, setattr, self.model.variables[-1], "type", "integer")

        def test_add_integer_var(self):
            self.assertRaises(ValueError, self.interface.Variable,'int_var', lb=-13, ub=499., type='integer')
