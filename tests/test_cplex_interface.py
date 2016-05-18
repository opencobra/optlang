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
    from optlang import cplex_interface
    import cplex
    CplexSolverError = cplex.exceptions.CplexSolverError

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
    CONVEX_QP_PATH = os.path.join(os.path.dirname(__file__), 'data/qplib_3256.lp')
    NONCONVEX_QP_PATH = os.path.join(os.path.dirname(__file__), 'data/qplib_1832.lp')

    class VariableTestCase(unittest.TestCase):
        def setUp(self):
            self.var = Variable('test')

        def test_set_wrong_type_raises(self):
            self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            model.optimize()
            self.assertEqual(model.status, 'optimal')
            self.assertEqual(model.objective.value, 0.8739215069684305)
            print([var.primal for var in model.variables])
            for i, j in zip([var.primal for var in model.variables], [0.8739215069684306, -16.023526143167608, 16.023526143167604, -14.71613956874283, 14.71613956874283, 4.959984944574658, 4.959984944574657, 4.959984944574658, 3.1162689467973905e-29, 2.926716099010601e-29, 0.0, 0.0, -6.112235045340358e-30, -5.6659435396316186e-30, 0.0, -4.922925402711085e-29, 0.0, 9.282532599166613, 0.0, 6.00724957535033, 6.007249575350331, 6.00724957535033, -5.064375661482091, 1.7581774441067828, 0.0, 7.477381962160285, 0.0, 0.22346172933182767, 45.514009774517454, 8.39, 0.0, 6.007249575350331, 0.0, -4.541857463865631, 0.0, 5.064375661482091, 0.0, 0.0, 2.504309470368734, 0.0, 0.0, -22.809833310204958, 22.809833310204958, 7.477381962160285, 7.477381962160285, 1.1814980932459636, 1.496983757261567, -0.0, 0.0, 4.860861146496815, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.064375661482091, 0.0, 5.064375661482091, 0.0, 0.0, 1.496983757261567, 10.000000000000002, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -29.175827135565804, 43.598985311997524, 29.175827135565804, 0.0, 0.0, 0.0, -1.2332237321082153e-29, 3.2148950476847613, 38.53460965051542, 5.064375661482091, 0.0, -1.2812714099825612e-29, -1.1331887079263237e-29, 17.530865429786694, 0.0, 0.0, 0.0, 4.765319193197458, -4.765319193197457, 21.79949265599876, -21.79949265599876, -3.2148950476847613, 0.0, -2.281503094067127, 2.6784818505075303, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_get_dual(self):
            self.assertEqual(self.var.dual, None)
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            model.optimize()
            self.assertEqual(model.status, 'optimal')
            self.assertEqual(model.objective.value, 0.8739215069684305)
            print([var.dual for var in model.variables])
            #for i, j in zip([var.dual for var in model.variables], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.022916186593776235, 0.0, 0.0, 0.0, -0.03437427989066435, 0.0, -0.007638728864592075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.005092485909728057, 0.0, 0.0, 0.0, 0.0, -0.005092485909728046, 0.0, 0.0, -0.005092485909728045, 0.0, 0.0, 0.0, -0.0611098309167366, -0.005092485909728045, 0.0, -0.003819364432296033, -0.00509248590972805, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03946676580039239, 0.0, 0.0, -0.005092485909728042, -0.0, -0.0012731214774320113, 0.0, -0.0916647463751049, 0.0, 0.0, 0.0, -0.0, -0.04583237318755246, 0.0, 0.0, -0.0916647463751049, -0.005092485909728045, -0.07002168125876067, 0.0, -0.06874855978132867, -0.0012731214774320113, 0.0, 0.0, 0.0, -0.001273121477432006, -0.0038193644322960392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.040739887277824405, -0.04583237318755245, -0.0012731214774320163, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03437427989066435, 0.0, 0.0, -0.04837861614241648]):
            #    self.assertAlmostEqual(i, j)

        def test_setting_lower_bound_higher_than_upper_bound_raises(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            self.assertRaises(ValueError, setattr, model.variables[0], 'lb', 10000000000.)

        def test_setting_nonnumerical_bounds_raises(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            self.assertRaises(Exception, setattr, model.variables[0], 'lb', 'Chicken soup')

        def test_setting_bounds(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            var = model.variables[0]
            var.lb = 1
            self.assertEqual(var.lb, 1)
            model.update()
            self.assertEqual(model.problem.variables.get_lower_bounds(var.name), 1)
            var.ub = 2
            self.assertEqual(var.ub, 2)
            model.update()
            self.assertEqual(model.problem.variables.get_upper_bounds(var.name), 2)

    class ConstraintTestCase(unittest.TestCase):
        def setUp(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            self.model = Model(problem=problem)
            self.constraint = Constraint(Variable('chip') + Variable('chap'), name='woodchips', lb=100)

        def test_get_primal(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertEqual(self.model.objective.value, 0.8739215069684305)
            print([constraint.primal for constraint in self.model.constraints])
            for i, j in zip([constraint.primal for constraint in self.model.constraints], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.048900234729145e-15, 0.0, 0.0, 0.0, -3.55971196577979e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5546369406238147e-17, 0.0, -5.080374405378186e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_get_dual(self):
            self.assertEqual(self.constraint.dual, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertEqual(self.model.objective.value, 0.8739215069684305)
            print([constraint.dual for constraint in self.model.constraints])
            #for i, j in zip([constraint.dual for constraint in self.model.constraints], [-0.047105494664984454, -0.042013008755256404, -0.042013008755256404, -0.09166474637510487, -0.09039162489767286, -0.02418930807120824, -0.022916186593776228, -0.034374279890664335, -0.034374279890664335, -0.028008672503504275, -0.07129480273619268, -0.029281793980936287, 0.005092485909728047, -0.06238295239416859, -0.06110983091673658, 0.010184971819456094, -0.0, -0.07129480273619268, -0.0, 0.0, -0.0, -0.0521979805747125, -0.06747543830389663, -0.0407398872778244, -0.039466765800392385, -0.09803035376226493, -0.104395961149425, 0.0, 0.0, -0.09166474637510488, -0.04837861614241646, -0.045832373187552435, -0.0521979805747125, -0.09803035376226493, -0.09166474637510488, -0.07511416716848872, -0.07002168125876067, -0.07002168125876067, -0.06874855978132866, -0.019096822161480172, -0.0, 0.0, 0.001273121477432012, 0.0, -0.07129480273619268, -0.042013008755256404, -0.04073988727782439, -0.04837861614241646, -0.045832373187552435, 0.007638728864592072, -0.0, 0.008911850342024089, -0.0, -0.0, 0.0, -0.0, 0.0, -0.042013008755256404, -0.042013008755256404, -0.001273121477432012, 0.0, -0.03564740136809635, -0.034374279890664335, 0.002546242954864024, -0.0, -0.08275289603308078, -0.08275289603308078, -0.11330781149144906, -0.050924859097280485, -0.04837861614241646, -0.054744223529576516, -0.08275289603308078]):
            #    self.assertAlmostEqual(i, j)

        def test_change_constraint_name(self):
            constraint = copy.copy(self.constraint)
            self.assertEqual(constraint.name, 'woodchips')
            constraint.name = 'ketchup'
            self.assertEqual(constraint.name, 'ketchup')
            self.assertEqual([constraint.name for constraint in self.model.constraints], ['M_13dpg_c', 'M_2pg_c', 'M_3pg_c', 'M_6pgc_c', 'M_6pgl_c', 'M_ac_c', 'M_ac_e', 'M_acald_c', 'M_acald_e', 'M_accoa_c', 'M_acon_C_c', 'M_actp_c', 'M_adp_c', 'M_akg_c', 'M_akg_e', 'M_amp_c', 'M_atp_c', 'M_cit_c', 'M_co2_c', 'M_co2_e', 'M_coa_c', 'M_dhap_c', 'M_e4p_c', 'M_etoh_c', 'M_etoh_e', 'M_f6p_c', 'M_fdp_c', 'M_for_c', 'M_for_e', 'M_fru_e', 'M_fum_c', 'M_fum_e', 'M_g3p_c', 'M_g6p_c', 'M_glc_D_e', 'M_gln_L_c', 'M_gln_L_e', 'M_glu_L_c', 'M_glu_L_e', 'M_glx_c', 'M_h2o_c', 'M_h2o_e', 'M_h_c', 'M_h_e', 'M_icit_c', 'M_lac_D_c', 'M_lac_D_e', 'M_mal_L_c', 'M_mal_L_e', 'M_nad_c', 'M_nadh_c', 'M_nadp_c', 'M_nadph_c', 'M_nh4_c', 'M_nh4_e', 'M_o2_c', 'M_o2_e', 'M_oaa_c', 'M_pep_c', 'M_pi_c', 'M_pi_e', 'M_pyr_c', 'M_pyr_e', 'M_q8_c', 'M_q8h2_c', 'M_r5p_c', 'M_ru5p_D_c', 'M_s7p_c', 'M_succ_c', 'M_succ_e', 'M_succoa_c', 'M_xu5p_D_c'])
            for i, constraint in enumerate(self.model.constraints):
                constraint.name = 'c'+ str(i)
            self.assertEqual([constraint.name for constraint in self.model.constraints], ['c' + str(i) for i in range(0, len(self.model.constraints))])

        def test_setting_lower_bound_higher_than_upper_bound_raises(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            print(model.constraints[0].lb)
            print(model.constraints[0].ub)
            self.assertRaises(ValueError, setattr, model.constraints[0], 'lb', 10000000000.)

        def test_setting_nonnumerical_bounds_raises(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            self.assertRaises(Exception, setattr, model.constraints[0], 'lb', 'Chicken soup')

        def test_setting_bounds(self):
            constraint = self.model.constraints[0]
            value = 42
            constraint.ub = value
            self.assertEqual(constraint.ub, value)
            constraint.lb = value
            self.assertEqual(constraint.lb, value)
            self.assertEqual(self.model.problem.linear_constraints.get_senses(constraint.name), "E")
            self.assertEqual(self.model.problem.linear_constraints.get_range_values(constraint.name), 0)


    class ObjectiveTestCase(unittest.TestCase):
        def setUp(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            self.model = Model(problem=problem)
            self.obj = self.model.objective

        def test_change_direction(self):
            self.obj.direction = "min"
            self.assertEqual(self.obj.direction, "min")
            self.assertEqual(self.model.problem.objective.get_sense(), self.model.problem.objective.sense.minimize)

            self.obj.direction = "max"
            self.assertEqual(self.obj.direction, "max")
            self.assertEqual(self.model.problem.objective.get_sense(), self.model.problem.objective.sense.maximize)


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
            self.model.optimize()
            value = self.model.objective.value
            model_copy = copy.copy(self.model)
            self.assertNotEqual(id(self.model), id(model_copy))
            self.assertNotEqual(id(self.model.problem), id(model_copy.problem))
            model_copy.optimize()
            self.assertAlmostEqual(value, model_copy.objective.value)
            self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in model_copy.variables.values()],
                             [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
            self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in model_copy.constraints],
                             [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

        def test_deepcopy(self):
            self.model.optimize()
            value = self.model.objective.value
            model_copy = copy.deepcopy(self.model)
            self.assertNotEqual(id(self.model), id(model_copy))
            self.assertNotEqual(id(self.model.problem), id(model_copy.problem))
            model_copy.optimize()
            self.assertAlmostEqual(value, model_copy.objective.value)
            self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in model_copy.variables.values()],
                             [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
            self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in model_copy.constraints],
                             [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

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

        def test_config_gets_copied_too(self):
            self.assertEquals(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            model_copy = copy.copy(self.model)
            self.assertEquals(model_copy.configuration.verbosity, 3)

        def test_pickle_empty_model(self):
            model = Model()
            self.assertEquals(model.objective, None)
            self.assertEquals(len(model.variables), 0)
            self.assertEquals(len(model.constraints), 0)
            pickle_string = pickle.dumps(model)
            from_pickle = pickle.loads(pickle_string)
            self.assertEquals(from_pickle.objective, None)
            self.assertEquals(len(from_pickle.variables), 0)
            self.assertEquals(len(from_pickle.constraints), 0)

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
            var = Variable('int_var', lb=-13, ub=499., type='integer')
            self.model.add(var)
            self.assertEqual(self.model.variables['int_var'].type, 'integer')
            self.assertEqual(self.model.variables['int_var'].ub, 499.)
            self.assertEqual(self.model.variables['int_var'].lb, -13)

        def test_add_non_cplex_conform_variable(self):
            var = Variable('12x!!@#5_3', lb=-666, ub=666)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(var.name, self.model.problem.variables.get_names(len(self.model.variables) - 1))
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            print(repickled.variables)
            var_from_pickle = repickled.variables['12x!!@#5_3']
            # self.assertEqual(var_from_pickle.name, glp_get_col_name(repickled.problem, var_from_pickle.index))

        def test_remove_variable(self):
            var = self.model.variables.values()[0]
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.assertNotIn(var, self.model.variables.values())
            self.assertEqual(var.problem, None)

        def test_add_linear_constraints(self):
            x = Variable('x', type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0., ub=3, type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y + z, ub=100.33, lb=-300)
            constr4 = Constraint(77*x, lb=10, name='Mul_constraint')
            constr5 = Constraint(x, ub=-10, name='Only_var_constraint')
            constr6 = Constraint(3, ub=88., name='Number_constraint')
            self.model.add(constr1)
            self.model.update()
            self.model.add(constr2)
            self.model.update()
            self.model.add(constr3)
            self.model.update()
            self.model.add([constr4, constr5, constr6])
            self.model.update()
            self.assertIn(constr1.name, self.model.constraints)
            self.assertIn(constr2.name, self.model.constraints)
            self.assertIn(constr3.name, self.model.constraints)
            self.assertIn(constr4.name, self.model.constraints)
            self.assertIn(constr5.name, self.model.constraints)
            self.assertIn(constr6.name, self.model.constraints)
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients((('test', 'y'), ('test', 'z'), ('test', 'x'))), [0.4, 66, 0.3])
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients((('test2', 'y'), ('test2', 'x'))), [1., 2.333])
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients('Mul_constraint', 'x'), 77.)
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients('Only_var_constraint', 'x'), 1.)

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
            self.assertIn(constr1, self.model.constraints)
            self.assertIn(constr2, self.model.constraints)
            self.assertIn(constr3, self.model.constraints)
            cplex_lines = [line.strip() for line in str(self.model).split('\n')]
            self.assertIn('test:       0.4 y + 66 z + 0.3 x - Rgtest  = -100', cplex_lines)
            self.assertIn('test2:      y + 2.333 x <= 96.997', cplex_lines)
            # Dummy_21:   y + z + 2.333 x - RgDummy_21  = -300
            self.assertRegexpMatches(str(self.model), '\s*Dummy_\d+:\s*y \+ z \+ 2\.333 x - .*  = -300')
            print(self.model)

        def test_remove_constraints(self):
            x = Variable('x', type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=4, ub=4, type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
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
            x = Variable('x', type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=3, ub=3, type='integer')
            constraint = Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
            self.model.add(constraint)
            self.assertRaises(ValueError, self.model.update)

        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x')
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y')]), [0.3, 0.4])
            z = Variable('z', lb=3, ub=4, type='integer')
            constraint += 77. * z
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y'), ('test', 'z')]), [0.3, 0.4, 77.])
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')
            print(self.model)

        def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            z = Variable('z', lb=2, ub=5, type='integer')
            constraint += 77. * z
            self.model.remove(constraint)
            self.assertEqual(constraint.__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')

        def test_change_of_objective_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            objective = Objective(0.3 * x + 0.4 * y, name='obj', direction='max')
            self.model.objective = objective
            for variable in self.model.variables:
                coeff = self.model.problem.objective.get_linear(variable.name)
                if variable.name == 'x':
                    self.assertEqual(coeff, 0.3)
                elif variable.name == 'y':
                    self.assertEqual(coeff, 0.4)
                else:
                    self.assertEqual(coeff, 0.)
            z = Variable('z', lb=0.000003, ub=0.000003, type='continuous')
            objective += 77. * z
            for variable in self.model.variables:
                coeff = self.model.problem.objective.get_linear(variable.name)
                if variable.name == 'x':
                    self.assertEqual(coeff, 0.3)
                elif variable.name == 'y':
                    self.assertEqual(coeff, 0.4)
                elif variable.name == 'z':
                    self.assertEqual(coeff, 77.)
                else:
                    self.assertEqual(coeff, 0.)

        def test_timeout(self):
            self.model.configuration.timeout = 0
            status = self.model.optimize()
            self.assertEqual(status, 'time_limit')

        def test_set_linear_objective_term(self):
            self.model._set_linear_objective_term(self.model.variables.R_TPI, 666.)
            self.assertEqual(self.model.problem.objective.get_linear(self.model.variables.R_TPI.name), 666.)

        def test__set_coefficients_low_level(self):
            constraint = self.model.constraints.M_atp_c
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], -59.8100000000000)
            constraint._set_coefficients_low_level({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], 666.)

        def test_primal_values(self):
            self.model.optimize()
            for k, v in self.model.primal_values.items():
                self.assertEquals(v, self.model.variables[k].primal)

        def test_reduced_costs(self):
            self.model.optimize()
            for k, v in self.model.reduced_costs.items():
                self.assertEquals(v, self.model.variables[k].dual)

        def test_dual_values(self):
            self.model.optimize()
            for k, v in self.model.dual_values.items():
                self.assertEquals(v, self.model.constraints[k].primal)

        def test_shadow_prices(self):
            self.model.optimize()
            for k, v in self.model.shadow_prices.items():
                self.assertEquals(v, self.model.constraints[k].dual)


    class ConfigurationTestCase(unittest.TestCase):
        def setUp(self):
            self.model = Model()
            self.configuration = self.model.configuration

        def test_lp_method(self):
            for option in cplex_interface._LP_METHODS:
                self.configuration.lp_method = option
                self.assertEqual(self.configuration.lp_method, option)
                self.assertEqual(self.model.problem.parameters.lpmethod.get(), getattr(self.model.problem.parameters.lpmethod.values, option))

            self.assertRaises(ValueError, setattr, self.configuration, "lp_method", "weird_stuff")

        def test_qp_method(self):
            for option in cplex_interface._QP_METHODS:
                self.configuration.qp_method = option
                self.assertEqual(self.configuration.qp_method, option)
                self.assertEqual(self.model.problem.parameters.qpmethod.get(), getattr(self.model.problem.parameters.qpmethod.values, option))

            self.assertRaises(ValueError, setattr, self.configuration, "qp_method", "weird_stuff")

        def test_solution_method(self):
            for option in cplex_interface._SOLUTION_TARGETS:
                self.configuration.solution_target = option
                self.assertEqual(self.configuration.solution_target, option)
                self.assertEqual(self.model.problem.parameters.solutiontarget.get(), cplex_interface._SOLUTION_TARGETS.index(option))

            self.assertRaises(ValueError, setattr, self.configuration, "solution_target", "weird_stuff")


    class QuadraticProgrammingTestCase(unittest.TestCase):
        def setUp(self):
            self.model = Model()
            self.x1 = Variable("x1", lb=0)
            self.x2 = Variable("x2", lb=0)
            self.c1 = Constraint(self.x1 + self.x2, lb=1)
            self.model.add([self.x1, self.x2, self.c1])

        def test_convex_obj(self):
            model = self.model
            obj = Objective(self.x1**2 + self.x2**2, direction="min")
            model.objective = obj
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.5)
            self.assertAlmostEqual(self.x1.primal, 0.5)
            self.assertAlmostEqual(self.x2.primal, 0.5)

            obj_2 = Objective(self.x1, direction="min")
            model.objective = obj_2
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.0)
            self.assertAlmostEqual(self.x1.primal, 0.0)
            self.assertGreaterEqual(self.x2.primal, 1.0)

        def test_non_convex_obj(self):
            model = self.model
            obj = Objective(self.x1 * self.x2, direction="min")
            model.objective = obj
            model.configuration.solution_target = "convex"
            self.assertRaises(CplexSolverError, model.optimize)
            model.configuration.solution_target = "local"
            model.configuration.qp_method = "barrier"
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0)
            model.configuration.solution_target = "global"
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0)

            obj_2 = Objective(self.x1, direction="min")
            model.objective = obj_2
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.0)
            self.assertAlmostEqual(self.x1.primal, 0.0)
            self.assertGreaterEqual(self.x2.primal, 1.0)

        def test_qp_convex(self):
            problem = cplex.Cplex()
            problem.read(CONVEX_QP_PATH)
            model = Model(problem=problem)
            self.assertEqual(len(model.variables), 651)
            self.assertEqual(len(model.constraints), 501)
            for constraint in model.constraints:
                self.assertTrue(constraint.is_Linear, "%s should be linear" % (str(constraint.expression)))
                self.assertFalse(constraint.is_Quadratic, "%s should not be quadratic" % (str(constraint.expression)))

            self.assertTrue(model.objective.is_Quadratic, "objective should be quadratic")
            self.assertFalse(model.objective.is_Linear, "objective should not be linear")

            model.optimize()
            self.assertAlmostEqual(model.objective.value, 32.2291282)

        @unittest.skip("Solving this is slow")
        def test_qp_non_convex(self):
            problem = cplex.Cplex()
            problem.read(NONCONVEX_QP_PATH)
            model = Model(problem=problem)
            self.assertEqual(len(model.variables), 31)
            self.assertEqual(len(model.constraints), 1)
            for constraint in model.constraints:
                self.assertTrue(constraint.is_Linear, "%s should be linear" % (str(constraint.expression)))
                self.assertFalse(constraint.is_Quadratic, "%s should not be quadratic" % (str(constraint.expression)))

            self.assertTrue(model.objective.is_Quadratic, "objective should be quadratic")
            self.assertFalse(model.objective.is_Linear, "objective should not be linear")

            model.configuration.solution_target = "convex"
            self.assertRaises(CplexSolverError, model.optimize)

            model.configuration.solution_target = "global"
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 2441.999999971)

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
