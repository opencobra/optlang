# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import copy
import os
import pickle
import random
import unittest

try:  # noqa: C901
    import cplex
except ImportError as e:

    if str(e).find('cplex') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise
else:

    import nose
    from optlang import interface
    from optlang.tests import abstract_test_cases

    from optlang.cplex_interface import Variable, Constraint, Model, Objective
    from optlang import cplex_interface

    CplexSolverError = cplex.exceptions.CplexSolverError

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
    TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')
    CONVEX_QP_PATH = os.path.join(os.path.dirname(__file__), 'data/qplib_3256.lp')
    NONCONVEX_QP_PATH = os.path.join(os.path.dirname(__file__), 'data/qplib_1832.lp')


    class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
        __test__ = True

        interface = cplex_interface

        # def setUp(self):
        #     self.var = Variable('test')
        #     self.model = Model()

        # def test_set_wrong_type_raises(self):
        #     self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
        #     self.model.add(self.var)
        #     self.model.update()
        #     self.assertRaises(ValueError, setattr, self.var, "type", "mustard")
        #     self.var.type = "integer"
        #     self.assertEqual(self.var.type, "integer")

        # def test_change_name(self):
        #     self.model.add(self.var)
        #     self.model.update()
        #     self.var.name = "test_2"
        #     self.assertEqual(self.var.name, "test_2")

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            model.optimize()
            self.assertEqual(model.status, 'optimal')
            self.assertEqual(model.objective.value, 0.8739215069684305)
            print([var.primal for var in model.variables])
            for i, j in zip([var.primal for var in model.variables],
                            [0.8739215069684306, -16.023526143167608, 16.023526143167604, -14.71613956874283,
                             14.71613956874283, 4.959984944574658, 4.959984944574657, 4.959984944574658,
                             3.1162689467973905e-29, 2.926716099010601e-29, 0.0, 0.0, -6.112235045340358e-30,
                             -5.6659435396316186e-30, 0.0, -4.922925402711085e-29, 0.0, 9.282532599166613, 0.0,
                             6.00724957535033, 6.007249575350331, 6.00724957535033, -5.064375661482091,
                             1.7581774441067828, 0.0, 7.477381962160285, 0.0, 0.22346172933182767, 45.514009774517454,
                             8.39, 0.0, 6.007249575350331, 0.0, -4.541857463865631, 0.0, 5.064375661482091, 0.0, 0.0,
                             2.504309470368734, 0.0, 0.0, -22.809833310204958, 22.809833310204958, 7.477381962160285,
                             7.477381962160285, 1.1814980932459636, 1.496983757261567, -0.0, 0.0, 4.860861146496815,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.064375661482091, 0.0, 5.064375661482091, 0.0, 0.0,
                             1.496983757261567, 10.000000000000002, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -29.175827135565804,
                             43.598985311997524, 29.175827135565804, 0.0, 0.0, 0.0, -1.2332237321082153e-29,
                             3.2148950476847613, 38.53460965051542, 5.064375661482091, 0.0, -1.2812714099825612e-29,
                             -1.1331887079263237e-29, 17.530865429786694, 0.0, 0.0, 0.0, 4.765319193197458,
                             -4.765319193197457, 21.79949265599876, -21.79949265599876, -3.2148950476847613, 0.0,
                             -2.281503094067127, 2.6784818505075303, 0.0]):
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
            self.var.type = "integer"
            model.add(self.var)
            model.optimize()
            self.assertTrue(self.var.dual is None)  # Cannot find reduced cost for MILP

        # def test_setting_lower_bound_higher_than_upper_bound_raises(self):
        #     problem = cplex.Cplex()
        #     problem.read(TESTMODELPATH)
        #     model = Model(problem=problem)
        #     self.assertRaises(ValueError, setattr, model.variables[0], 'lb', 10000000000.)

        def test_changing_variable_names_is_reflected_in_the_solver(self):
            model = Model(problem=cplex.Cplex(TESTMODELPATH))
            for i, variable in enumerate(model.variables):
                variable.name = "var" + str(i)
                self.assertEqual(variable.name, "var" + str(i))
                self.assertEqual(model.problem.variables.get_names(i), "var" + str(i))

        # def test_setting_nonnumerical_bounds_raises(self):
        #     problem = cplex.Cplex()
        #     problem.read(TESTMODELPATH)
        #     model = Model(problem=problem)
        #     self.assertRaises(Exception, setattr, model.variables[0], 'lb', 'Chicken soup')

        def test_cplex_setting_bounds(self):
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

        # def test_set_bounds_to_none(self):
        #     model = Model()
        #     var = Variable("test_var")
        #     obj = Objective(var)
        #     model.objective = obj
        #     self.assertEqual(model.optimize(), interface.UNBOUNDED)
        #     var.ub = 10
        #     self.assertEqual(model.optimize(), interface.OPTIMAL)
        #     var.ub = None
        #     self.assertEqual(model.optimize(), interface.UNBOUNDED)
        #     obj.direction = "min"
        #     var.lb = -10
        #     self.assertEqual(model.optimize(), interface.OPTIMAL)
        #     var.lb = None
        #     self.assertEqual(model.optimize(), interface.UNBOUNDED)


    class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
        def setUp(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            self.model = Model(problem=problem)
            self.constraint = Constraint(Variable('chip') + Variable('chap'), name='woodchips', lb=100)

        def test_set_linear_coefficients(self):
            self.model.add(self.constraint)
            self.constraint.set_linear_coefficients({Variable('chip'): 33., self.model.variables.R_PGK: -33})
            sparse_pair = self.model.problem.linear_constraints.get_rows(self.constraint.name)
            self.assertEqual(dict(zip(self.model.problem.variables.get_names(sparse_pair.ind), sparse_pair.val)),
                             dict([('R_PGK', -33.0), ('chap', 1.0), ('chip', 33.0)]))

        def test_indicator_constraint_support(self):
            Constraint(
                Variable('chip_2'),
                indicator_variable=Variable('chip', type='binary'), active_when=0, lb=0,
                ub=0,
                name='indicator_constraint_fwd_1'
            )

        def test_get_primal(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertEqual(self.model.objective.value, 0.8739215069684305)
            print([constraint.primal for constraint in self.model.constraints])
            for i, j in zip([constraint.primal for constraint in self.model.constraints],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             4.048900234729145e-15, 0.0, 0.0, 0.0, -3.55971196577979e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 2.5546369406238147e-17, 0.0, -5.080374405378186e-29, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_get_dual(self):
            self.assertEqual(self.constraint.dual, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertEqual(self.model.objective.value, 0.8739215069684305)
            print([constraint.dual for constraint in self.model.constraints])

        def test_change_constraint_name(self):
            constraint = copy.copy(self.constraint)
            self.assertEqual(constraint.name, 'woodchips')
            constraint.name = 'ketchup'
            self.assertEqual(constraint.name, 'ketchup')
            self.assertEqual([constraint.name for constraint in self.model.constraints],
                             ['M_13dpg_c', 'M_2pg_c', 'M_3pg_c', 'M_6pgc_c', 'M_6pgl_c', 'M_ac_c', 'M_ac_e',
                              'M_acald_c', 'M_acald_e', 'M_accoa_c', 'M_acon_C_c', 'M_actp_c', 'M_adp_c', 'M_akg_c',
                              'M_akg_e', 'M_amp_c', 'M_atp_c', 'M_cit_c', 'M_co2_c', 'M_co2_e', 'M_coa_c', 'M_dhap_c',
                              'M_e4p_c', 'M_etoh_c', 'M_etoh_e', 'M_f6p_c', 'M_fdp_c', 'M_for_c', 'M_for_e', 'M_fru_e',
                              'M_fum_c', 'M_fum_e', 'M_g3p_c', 'M_g6p_c', 'M_glc_D_e', 'M_gln_L_c', 'M_gln_L_e',
                              'M_glu_L_c', 'M_glu_L_e', 'M_glx_c', 'M_h2o_c', 'M_h2o_e', 'M_h_c', 'M_h_e', 'M_icit_c',
                              'M_lac_D_c', 'M_lac_D_e', 'M_mal_L_c', 'M_mal_L_e', 'M_nad_c', 'M_nadh_c', 'M_nadp_c',
                              'M_nadph_c', 'M_nh4_c', 'M_nh4_e', 'M_o2_c', 'M_o2_e', 'M_oaa_c', 'M_pep_c', 'M_pi_c',
                              'M_pi_e', 'M_pyr_c', 'M_pyr_e', 'M_q8_c', 'M_q8h2_c', 'M_r5p_c', 'M_ru5p_D_c', 'M_s7p_c',
                              'M_succ_c', 'M_succ_e', 'M_succoa_c', 'M_xu5p_D_c'])
            for i, constraint in enumerate(self.model.constraints):
                constraint.name = 'c' + str(i)
            self.assertEqual([constraint.name for constraint in self.model.constraints],
                             ['c' + str(i) for i in range(0, len(self.model.constraints))])

        def test_setting_lower_bound_higher_than_upper_bound_raises(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            print(model.constraints[0].lb)
            print(model.constraints[0].ub)
            self.assertRaises(ValueError, setattr, model.constraints[0], 'lb', 10000000000.)
            self.assertRaises(ValueError, setattr, model.constraints[0], "ub", -1000000000.)

            self.assertRaises(ValueError, Constraint, 0, lb=0, ub=-1)

        def test_setting_nonnumerical_bounds_raises(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            self.assertRaises(Exception, setattr, model.constraints[0], 'lb', 'Chicken soup')

        def test_set_constraint_bounds_to_none(self):
            model = Model()
            var = Variable("test")
            const = Constraint(var, lb=-10, ub=10)
            obj = Objective(var)
            model.add(const)
            model.objective = obj
            self.assertEqual(model.optimize(), interface.OPTIMAL)
            const.ub = None
            self.assertEqual(model.optimize(), interface.UNBOUNDED)
            const.ub = 10
            const.lb = None
            obj.direction = "min"
            self.assertEqual(model.optimize(), interface.UNBOUNDED)
            const.lb = -10
            self.assertEqual(model.optimize(), interface.OPTIMAL)


    class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
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


    class ModelTestCase(abstract_test_cases.AbstractModelTestCase):

        def setUp(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            assert problem.variables.get_num() > 0
            self.model = Model(problem=problem)

        def test_create_empty_model(self):
            model = Model()
            self.assertEqual(len(model.constraints), 0)
            self.assertEqual(len(model.variables), 0)
            self.assertEqual(model.objective.expression, 0)

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

        def test_pickle_empty_model(self):
            model = Model()
            self.assertEquals(model.objective.expression, 0)
            self.assertEquals(len(model.variables), 0)
            self.assertEquals(len(model.constraints), 0)
            pickle_string = pickle.dumps(model)
            from_pickle = pickle.loads(pickle_string)
            self.assertEquals(from_pickle.objective.expression, 0)
            self.assertEquals(len(from_pickle.variables), 0)
            self.assertEquals(len(from_pickle.constraints), 0)

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

        def test_config_gets_copied_too(self):
            self.assertEquals(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            model_copy = copy.copy(self.model)
            self.assertEquals(model_copy.configuration.verbosity, 3)

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
            self.assertEqual(var_from_pickle.name, self.model.problem.variables.get_names()[-1])

        def test_remove_variable(self):
            var = self.model.variables.values()[0]
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.assertNotIn(var, self.model.variables.values())
            self.assertEqual(var.problem, None)

        def test_remove_variable_str(self):
            var = self.model.variables.values()[0]
            self.model.remove(var.name)
            self.assertNotIn(var, self.model.variables.values())
            self.assertNotIn(var.name, self.model.problem.variables.get_names())
            self.assertEqual(var.problem, None)

        def test_add_constraints(self):
            x = Variable('x', type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0., ub=3, type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y + z, ub=100.33, lb=-300)
            constr4 = Constraint(77 * x, lb=10, name='Mul_constraint')
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
            self.assertEqual(
                self.model.problem.linear_constraints.get_coefficients((('test', 'y'), ('test', 'z'), ('test', 'x'))),
                [0.4, 66, 0.3])
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients((('test2', 'y'), ('test2', 'x'))),
                             [1., 2.333])
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients('Mul_constraint', 'x'), 77.)
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients('Only_var_constraint', 'x'), 1.)

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

        @unittest.skip('Skipping for now')
        def test_add_quadratic_constraints(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constr1 = Constraint(0.3 * x * y + 0.4 * y ** 2 + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y ** 2 + z + 33, ub=100.33, lb=-300)
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
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y')]),
                             [0.3, 0.4])
            z = Variable('z', lb=3, ub=4, type='integer')
            constraint += 77. * z
            self.assertEqual(
                self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y'), ('test', 'z')]),
                [0.3, 0.4, 77.])
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

        def test_change_variable_bounds(self):
            inner_prob = self.model.problem
            inner_problem_bounds = list(
                zip(inner_prob.variables.get_lower_bounds(), inner_prob.variables.get_upper_bounds())
            )
            bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertEqual(bounds, inner_problem_bounds)
            for var in self.model.variables.values():
                var.lb = random.uniform(-1000, 1000)
                var.ub = random.uniform(var.lb, 1000)
            self.model.update()
            inner_problem_bounds_new = list(
                zip(inner_prob.variables.get_lower_bounds(), inner_prob.variables.get_upper_bounds())
            )
            bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        def test_change_variable_type(self):
            for variable in self.model.variables:
                variable.type = 'integer'
            # There seems to be a bug in the get_types function TODO Fix this
            # self.assertEqual(set(self.model.problem.variables.get_types()), {'I'})

        def test_change_constraint_bounds(self):
            constraint = self.model.constraints[0]
            value = 42
            constraint.ub = value
            self.assertEqual(constraint.ub, value)
            constraint.lb = value
            self.assertEqual(constraint.lb, value)
            self.assertEqual(self.model.problem.linear_constraints.get_senses(constraint.name), "E")
            self.assertEqual(self.model.problem.linear_constraints.get_range_values(constraint.name), 0)

        def test_initial_objective(self):
            self.assertEqual(self.model.objective.expression.__str__(), '1.0*R_Biomass_Ecoli_core_w_GAM')

        def test_optimize(self):
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertAlmostEqual(self.model.objective.value, 0.8739215069684303)

        def test_optimize_milp(self):
            problem = cplex.Cplex(TESTMILPMODELPATH)
            milp_model = Model(problem=problem)
            milp_model.optimize()
            self.assertEqual(milp_model.status, 'optimal')
            self.assertAlmostEqual(milp_model.objective.value, 122.5)
            for variable in milp_model.variables:
                if variable.type == 'integer':
                    self.assertEqual(variable.primal % 1, 0)

        def test_change_objective(self):
            """Test that all different kinds of linear objective specification work."""
            print(self.model.variables.values()[0:2])
            v1, v2 = self.model.variables.values()[0:2]
            self.model.objective = Objective(1. * v1 + 1. * v2)
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_PGK + 1.0*R_Biomass_Ecoli_core_w_GAM')
            self.model.objective = Objective(v1 + v2)
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_PGK + 1.0*R_Biomass_Ecoli_core_w_GAM')

        def test_number_objective(self):
            self.model.objective = Objective(0.)
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n0')
            self.assertEqual(set(self.model.problem.objective.get_linear()), {0.})

        def test_raise_on_non_linear_objective(self):
            """Test that an exception is raised when a non-linear objective is added to the model."""
            v1, v2 = self.model.variables.values()[0:2]
            self.assertRaises(ValueError, Objective, v1 * v2 ** 3)

        def test_iadd_objective(self):
            v2, v3 = self.model.variables.values()[1:3]
            self.model.objective += 2. * v2 - 3. * v3
            obj_coeff = self.model.problem.objective.get_linear()
            self.assertEqual(obj_coeff,
                             [1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0]
                             )

        def test_imul_objective(self):
            self.model.objective *= 2.
            obj_coeff = self.model.problem.objective.get_linear()
            print(obj_coeff)
            self.assertEqual(obj_coeff,
                             [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0]
                             )

        def test_set_copied_objective(self):
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_Biomass_Ecoli_core_w_GAM')

        def test_timeout(self):
            self.model.configuration.timeout = 0
            status = self.model.optimize()
            self.assertEqual(status, 'time_limit')

        def test_set_linear_coefficients_objective(self):
            self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
            self.assertEqual(self.model.problem.objective.get_linear(self.model.variables.R_TPI.name), 666.)

        def test_instantiating_model_with_different_solver_problem_raises(self):
            self.assertRaises(TypeError, Model, problem='Chicken soup')

        def test_set_linear_coefficients_constraint(self):
            constraint = self.model.constraints.M_atp_c
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], -59.8100000000000)
            constraint.set_linear_coefficients({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
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

        def test_change_objective_can_handle_removed_vars(self):
            self.model.objective = Objective(self.model.variables[0])
            self.model.remove(self.model.variables[0])
            self.model.update()
            self.model.objective = Objective(self.model.variables[1] ** 2)
            self.model.remove(self.model.variables[1])
            self.model.objective = Objective(self.model.variables[2])

        def test_clone_model(self):
            self.assertEquals(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            cloned_model = Model.clone(self.model)
            self.assertEquals(cloned_model.configuration.verbosity, 3)
            self.assertEquals(len(cloned_model.variables), len(self.model.variables))
            self.assertEquals(len(cloned_model.constraints), len(self.model.constraints))


    class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):
        def setUp(self):
            self.model = Model()
            self.configuration = self.model.configuration

        def test_lp_method(self):
            for option in cplex_interface._LP_METHODS:
                self.configuration.lp_method = option
                self.assertEqual(self.configuration.lp_method, option)
                self.assertEqual(self.model.problem.parameters.lpmethod.get(),
                                 getattr(self.model.problem.parameters.lpmethod.values, option))

            self.assertRaises(ValueError, setattr, self.configuration, "lp_method", "weird_stuff")

        def test_qp_method(self):
            for option in cplex_interface._QP_METHODS:
                self.configuration.qp_method = option
                self.assertEqual(self.configuration.qp_method, option)
                self.assertEqual(self.model.problem.parameters.qpmethod.get(),
                                 getattr(self.model.problem.parameters.qpmethod.values, option))

            self.assertRaises(ValueError, setattr, self.configuration, "qp_method", "weird_stuff")
            self.configuration.solution_target = None
            self.assertEqual(self.model.problem.parameters.solutiontarget.get(),
                             self.model.problem.parameters.solutiontarget.default())

        def test_solution_method(self):
            for option in cplex_interface._SOLUTION_TARGETS:
                self.configuration.solution_target = option
                self.assertEqual(self.configuration.solution_target, option)
                self.assertEqual(self.model.problem.parameters.solutiontarget.get(),
                                 cplex_interface._SOLUTION_TARGETS.index(option))

            self.assertRaises(ValueError, setattr, self.configuration, "solution_target", "weird_stuff")

        def test_verbosity(self):
            for i in range(4):
                self.model.configuration.verbosity = i
                self.assertEqual(self.model.configuration.verbosity, i)
            self.assertRaises(ValueError, setattr, self.configuration, "verbosity", 8)

        def test_presolve(self):
            for presolve in (True, False, "auto"):
                self.configuration.presolve = presolve
                self.assertEqual(self.configuration.presolve, presolve)

            self.assertRaises(ValueError, setattr, self.configuration, "presolve", "trump")


    class QuadraticProgrammingTestCase(abstract_test_cases.AbstractQuadraticProgrammingTestCase):
        def setUp(self):
            self.model = Model()
            self.x1 = Variable("x1", lb=0)
            self.x2 = Variable("x2", lb=0)
            self.c1 = Constraint(self.x1 + self.x2, lb=1)
            self.model.add([self.x1, self.x2, self.c1])

        def test_convex_obj(self):
            model = self.model
            obj = Objective(self.x1 ** 2 + self.x2 ** 2, direction="min")
            print(obj.expression)
            print(obj.is_Quadratic)
            model.objective = obj
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.5)
            self.assertAlmostEqual(self.x1.primal, 0.5)
            self.assertAlmostEqual(self.x2.primal, 0.5)

            obj_2 = Objective(self.x1, direction="min")
            print(model.__str__()[0:1000])
            model.objective = obj_2
            print(model.__str__()[0:1000])
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

        @unittest.skip("")
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


if __name__ == '__main__':
    nose.runmodule()
