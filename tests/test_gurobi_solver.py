# Copyright (c) 2016 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import unittest

try:
    import gurobipy
except ImportError as e:

    class TestMissingDependency(unittest.TestCase):

        @unittest.skip('Missing dependency - ' + str(e))
        def test_fail(self):
            pass

else:

    import copy

    import random

    import os

    import nose
    import pickle

    from optlang.gurobi_interface import Variable, Constraint, Model, Objective

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
    TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')


    class VariableTestCase(unittest.TestCase):
        def setUp(self):
            self.var = Variable('test')

        def test_internal_variable(self):
            self.assertEqual(self.var._internal_variable, None)

        def test_set_wrong_type_raises(self):
            self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            model.optimize()
            for i, j in zip([var.primal for var in model.variables], [0.8739215069684306, -16.023526143167608, 16.023526143167604, -14.71613956874283, 14.71613956874283, 4.959984944574658, 4.959984944574657, 4.959984944574658, 3.1162689467973905e-29, 2.926716099010601e-29, 0.0, 0.0, -6.112235045340358e-30, -5.6659435396316186e-30, 0.0, -4.922925402711085e-29, 0.0, 9.282532599166613, 0.0, 6.00724957535033, 6.007249575350331, 6.00724957535033, -5.064375661482091, 1.7581774441067828, 0.0, 7.477381962160285, 0.0, 0.22346172933182767, 45.514009774517454, 8.39, 0.0, 6.007249575350331, 0.0, -4.541857463865631, 0.0, 5.064375661482091, 0.0, 0.0, 2.504309470368734, 0.0, 0.0, -22.809833310204958, 22.809833310204958, 7.477381962160285, 7.477381962160285, 1.1814980932459636, 1.496983757261567, -0.0, 0.0, 4.860861146496815, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.064375661482091, 0.0, 5.064375661482091, 0.0, 0.0, 1.496983757261567, 10.000000000000002, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -29.175827135565804, 43.598985311997524, 29.175827135565804, 0.0, 0.0, 0.0, -1.2332237321082153e-29, 3.2148950476847613, 38.53460965051542, 5.064375661482091, 0.0, -1.2812714099825612e-29, -1.1331887079263237e-29, 17.530865429786694, 0.0, 0.0, 0.0, 4.765319193197458, -4.765319193197457, 21.79949265599876, -21.79949265599876, -3.2148950476847613, 0.0, -2.281503094067127, 2.6784818505075303, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_get_dual(self):
            self.assertEqual(self.var.dual, None)
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            model.optimize()
            print([var.dual for var in model.variables])
            for i, j in zip([var.dual for var in model.variables], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.022916186593776214, 0.0, 0.0, 0.0, -0.03437427989066433, 0.0, -0.007638728864592076, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.005092485909728051, 0.0, 0.0, 0.0, 0.0, -0.005092485909728049, 0.0, 0.0, -0.005092485909728071, 0.0, 0.0, 0.0, -0.06110983091673658, -0.005092485909728054, 0.0, -0.003819364432296038, -0.005092485909728044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03946676580039238, 0.0, 0.0, -0.005092485909728051, 0.0, -0.0012731214774320122, 0.0, -0.09166474637510488, 0.0, 0.0, 0.0, -0.0, -0.045832373187552435, 0.0, 0.0, -0.09166474637510488, -0.005092485909728051, -0.07002168125876065, 0.0, -0.06874855978132864, -0.0012731214774320126, 0.0, 0.0, 0.0, -0.0012731214774320161, -0.003819364432296038, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04073988727782439, -0.045832373187552435, -0.0012731214774320083, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03437427989066433, 0.0, 0.0, -0.04837861614241646]):
                self.assertAlmostEqual(i, j)

        def test_setting_lower_bound_higher_than_upper_bound_raises(self):
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            self.assertRaises(ValueError, setattr, model.variables[0], 'lb', 10000000000.)

        def test_setting_nonnumerical_bounds_raises(self):
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            self.assertRaises(Exception, setattr, model.variables[0], 'lb', 'Chicken soup')

        def test_changing_variable_names_is_reflected_in_the_solver(self):
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            for i, variable in enumerate(model.variables):
                print(variable._internal_variable is not None)
                print(variable.problem.name)
                variable.name = "var"+str(i)
                print(variable.problem.name)
                print(variable.name)
                print(variable._internal_variable is not None)
                self.assertEqual(variable.name, "var"+str(i))
                self.assertEqual(variable._internal_variable.getAttr('VarName'), "var"+str(i))

        def test_setting_bounds(self):
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            var = model.variables[0]
            var.lb = 1
            self.assertEqual(var.lb, 1)
            model.problem.update()
            self.assertEqual(var._internal_variable.getAttr('LB'), 1)
            var.ub = 2
            self.assertEqual(var.ub, 2)
            model.problem.update()
            self.assertEqual(var._internal_variable.getAttr('UB'), 2)


    class ConstraintTestCase(unittest.TestCase):
        def setUp(self):
            self.model = Model(problem=gurobipy.read(TESTMODELPATH))
            self.constraint = Constraint(Variable('chip') + Variable('chap'), name='woodchips', lb=100)

        def test_get_primal(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            print([constraint.primal for constraint in self.model.constraints])
            for i, j in zip([constraint.primal for constraint in self.model.constraints], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.048900234729145e-15, 0.0, 0.0, 0.0, -3.55971196577979e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5546369406238147e-17, 0.0, -5.080374405378186e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_get_dual(self):
            self.assertEqual(self.constraint.dual, None)
            self.model.optimize()
            for i, j in zip([constraint.dual for constraint in self.model.constraints], [-0.04710549466498445, -0.0420130087552564, -0.0420130087552564, -0.09166474637510486, -0.09039162489767284, -0.024189308071208226, -0.022916186593776214, -0.03437427989066433, -0.03437427989066433, -0.028008672503504264, -0.07129480273619268, -0.029281793980936277, 0.0, -0.06238295239416859, -0.06110983091673658, 0.005092485909728049, -0.005092485909728049, -0.07129480273619268, -0.0, -0.0, 0.0, -0.0521979805747125, -0.06747543830389663, -0.04073988727782439, -0.03946676580039238, -0.09803035376226493, -0.104395961149425, -0.0, -0.0, -0.09166474637510488, -0.04837861614241646, -0.045832373187552435, -0.0521979805747125, -0.09803035376226493, -0.09166474637510488, -0.0751141671684887, -0.07002168125876065, -0.07002168125876065, -0.06874855978132864, -0.019096822161480183, -0.0, -0.0, 0.0012731214774320122, -0.0, -0.07129480273619268, -0.042013008755256404, -0.04073988727782439, -0.04837861614241646, -0.045832373187552435, 0.007638728864592073, 0.0, 0.008911850342024082, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0420130087552564, -0.0420130087552564, -0.0012731214774320122, -0.0, -0.03564740136809634, -0.03437427989066433, 0.0, -0.0025462429548640244, -0.08275289603308078, -0.08275289603308078, -0.11330781149144906, -0.050924859097280485, -0.04837861614241646, -0.054744223529576516, -0.08275289603308078]):
                self.assertAlmostEqual(i, j)

    #     def test_change_constraint_name(self):
    #         constraint = copy.copy(self.constraint)
    #         self.assertEqual(constraint.name, 'woodchips')
    #         constraint.name = 'ketchup'
    #         self.assertEqual(constraint.name, 'ketchup')
    #         self.assertEqual([constraint.name for constraint in self.model.constraints], ['M_13dpg_c', 'M_2pg_c', 'M_3pg_c', 'M_6pgc_c', 'M_6pgl_c', 'M_ac_c', 'M_ac_e', 'M_acald_c', 'M_acald_e', 'M_accoa_c', 'M_acon_C_c', 'M_actp_c', 'M_adp_c', 'M_akg_c', 'M_akg_e', 'M_amp_c', 'M_atp_c', 'M_cit_c', 'M_co2_c', 'M_co2_e', 'M_coa_c', 'M_dhap_c', 'M_e4p_c', 'M_etoh_c', 'M_etoh_e', 'M_f6p_c', 'M_fdp_c', 'M_for_c', 'M_for_e', 'M_fru_e', 'M_fum_c', 'M_fum_e', 'M_g3p_c', 'M_g6p_c', 'M_glc_D_e', 'M_gln_L_c', 'M_gln_L_e', 'M_glu_L_c', 'M_glu_L_e', 'M_glx_c', 'M_h2o_c', 'M_h2o_e', 'M_h_c', 'M_h_e', 'M_icit_c', 'M_lac_D_c', 'M_lac_D_e', 'M_mal_L_c', 'M_mal_L_e', 'M_nad_c', 'M_nadh_c', 'M_nadp_c', 'M_nadph_c', 'M_nh4_c', 'M_nh4_e', 'M_o2_c', 'M_o2_e', 'M_oaa_c', 'M_pep_c', 'M_pi_c', 'M_pi_e', 'M_pyr_c', 'M_pyr_e', 'M_q8_c', 'M_q8h2_c', 'M_r5p_c', 'M_ru5p_D_c', 'M_s7p_c', 'M_succ_c', 'M_succ_e', 'M_succoa_c', 'M_xu5p_D_c']
    # )
    #         for i, constraint in enumerate(self.model.constraints):
    #             constraint.name = 'c'+ str(i)
    #         self.assertEqual([constraint.name for constraint in self.model.constraints], ['c' + str(i) for i in range(0, len(self.model.constraints))])
    #
        def test_setting_lower_bound_higher_than_upper_bound_raises(self):
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            self.assertRaises(ValueError, setattr, model.constraints[0], 'lb', 10000000000.)
    #
    #     def test_setting_nonnumerical_bounds_raises(self):
    #         model = Model(problem=glpk_read_cplex(TESTMODELPATH))
    #         self.assertRaises(Exception, setattr, model.constraints[0], 'lb', 'Chicken soup')


    # class ObjectiveTestCase(unittest.TestCase):
    #     def setUp(self):
    #         self.model = Model(problem=glpk_read_cplex(TESTMODELPATH))
    #         self.obj = self.model.objective

        # def test_change_direction(self):
        #     self.obj.direction = "min"
        #     self.assertEqual(self.obj.direction, "min")
        #     self.assertEqual(glpk_interface.glp_get_obj_dir(self.model.problem), glpk_interface.GLP_MIN)
        #
        #     self.obj.direction = "max"
        #     self.assertEqual(self.obj.direction, "max")
        #     self.assertEqual(glpk_interface.glp_get_obj_dir(self.model.problem), glpk_interface.GLP_MAX)


    class SolverTestCase(unittest.TestCase):
        def setUp(self):
            # problem = gurobipy.Model()
            problem = gurobipy.read(TESTMODELPATH)
            self.model = Model(problem=problem)

        def test_create_empty_model(self):
            model = Model()
            self.assertEqual(model.problem.getAttr('NumVars'), 0)
            self.assertEqual(model.problem.getAttr('NumConstrs'), 0)
            self.assertEqual(model.name, None)
            self.assertEqual(model.problem.getAttr('ModelName'), '')
            model = Model(name="empty_problem")
            self.assertEqual(model.problem.getAttr('ModelName'), 'empty_problem')
        #
        # def test_pickle_ability(self):
        #     self.model.optimize()
        #     value = self.model.objective.value
        #     pickle_string = pickle.dumps(self.model)
        #     from_pickle = pickle.loads(pickle_string)
        #     from_pickle.optimize()
        #     self.assertAlmostEqual(value, from_pickle.objective.value)
        #     self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in from_pickle.variables.values()],
        #                      [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
        #     self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in from_pickle.constraints],
        #                      [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

        # def test_copy(self):
        #     model_copy = copy.copy(self.model)
        #     self.assertNotEqual(id(self.model), id(model_copy))
        #     self.assertEqual(id(self.model.problem), id(model_copy.problem))
        #
        # def test_deepcopy(self):
        #     model_copy = copy.deepcopy(self.model)
        #     self.assertNotEqual(id(self.model), id(model_copy))
        #     self.assertNotEqual(id(self.model.problem), id(model_copy.problem))

        # def test_config_gets_copied_too(self):
        #     self.assertEquals(self.model.configuration.verbosity, 0)
        #     self.model.configuration.verbosity = 3
        #     model_copy = copy.copy(self.model)
        #     self.assertEquals(model_copy.configuration.verbosity, 3)
        #
        def test_init_from_existing_problem(self):
            self.assertEqual(len(self.model.variables), len(self.model.problem.getVars()))
            self.assertEqual(len(self.model.constraints), len(self.model.problem.getConstrs()))
            self.assertEqual(self.model.variables.keys(),
                             [var.VarName for var in self.model.problem.getVars()])

            self.assertEqual(self.model.constraints.keys(),
                         [constr.ConstrName for constr in self.model.problem.getConstrs()])

        def test_add_variable(self):
            var = Variable('x')
            self.model.add(var)
            print(self.model._pending_modifications)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(self.model.variables.values().count(var), 1)
            self.assertEqual(self.model.variables['x'].problem, var.problem)
            print(var.name)
            print(self.model.problem.getVars())
            print(self.model._pending_modifications)
            self.model.update()
            print(self.model._pending_modifications)
            print(self.model.problem.getVars())
            self.assertEqual(self.model.problem.getVarByName(var.name).getAttr('VType'), gurobipy.GRB.CONTINUOUS)
            var = Variable('y', lb=-13)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.model.problem.update()
            self.assertEqual(self.model.problem.getVarByName(var.name).getAttr('VType'), gurobipy.GRB.CONTINUOUS)
            self.assertEqual(self.model.variables['x'].lb, None)
            self.assertEqual(self.model.variables['x'].ub, None)
            self.assertEqual(self.model.variables['y'].lb, -13)
            self.assertEqual(self.model.variables['x'].ub, None)
            var = Variable('x_with_ridiculously_long_variable_name_asdffffffffasdfasdfasdfasdfasdfasdfasdf')
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(self.model.variables.values().count(var), 1)
            # var = Variable('x_with_ridiculously_long_variable_name_asdffffffffasdfasdfasdfasdfasdfasdfasdf')
            # self.assertRaises(Exception, self.model.add, var)
            # self.assertEqual(len(self.model.variables), len(self.model.problem.getVars()))

        def test_add_integer_var(self):
            var = Variable('int_var', lb=-13, ub=500, type='integer')
            self.model.add(var)
            self.assertEqual(self.model.variables['int_var'].type, 'integer')
            self.assertEqual(self.model.problem.getVarByName(var.name).getAttr('VType'), gurobipy.GRB.INTEGER)
            self.assertEqual(self.model.variables['int_var'].ub, 500)
            self.assertEqual(self.model.variables['int_var'].lb, -13)

        def test_add_non_cplex_conform_variable(self):
            var = Variable('12x!!@#5_3', lb=-666, ub=666)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.model.problem.update()
            self.assertEqual(var.name, self.model.problem.getVarByName(var.name).VarName)
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            var_from_pickle = repickled.variables['12x!!@#5_3']
            self.assertEqual(var_from_pickle.name, repickled.problem.getVarByName(var.name).VarName)

        def test_remove_variable(self):
            var = self.model.variables.values()[0]
            self.assertEqual(self.model.constraints['M_atp_c'].__str__(),
                             'M_atp_c: 0.0 <= -1.0*R_ACKr - 1.0*R_ADK1 + 1.0*R_ATPS4r - 1.0*R_PGK - 1.0*R_SUCOAS - 59.81*R_Biomass_Ecoli_core_w_GAM - 1.0*R_GLNS - 1.0*R_GLNabc - 1.0*R_PFK - 1.0*R_PPCK - 1.0*R_PPS + 1.0*R_PYK - 1.0*R_ATPM <= 0.0')
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.model.problem.update()
            self.assertEqual(self.model.constraints['M_atp_c'].__str__(),
                             'M_atp_c: 0.0 <= -1.0*R_ACKr - 1.0*R_ADK1 + 1.0*R_ATPS4r - 1.0*R_PGK - 1.0*R_SUCOAS - 1.0*R_GLNS - 1.0*R_GLNabc - 1.0*R_PFK - 1.0*R_PPCK - 1.0*R_PPS + 1.0*R_PYK - 1.0*R_ATPM <= 0.0')
            self.assertNotIn(var, self.model.variables.values())
            self.assertEqual(self.model.problem.getVarByName(var.name), None)
            self.assertEqual(var.problem, None)

        def test_remove_variable_str(self):
            var = self.model.variables.values()[0]
            self.model.remove(var.name)
            self.model.problem.update()
            self.assertNotIn(var, self.model.variables.values())
            self.assertEqual(self.model.problem.getVarByName(var.name), None)
            self.assertEqual(var.problem, None)

        def test_add_constraints(self):
            x = Variable('x', lb=0, ub=1, type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0., ub=10., type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y + z, lb=-300)
            constr4 = Constraint(x, lb=-300, ub=-300)
            constr5 = Constraint(3*x)
            self.model.add(constr1)
            self.model.add(constr2)
            self.model.add(constr3)
            self.model.add(constr4)
            self.model.problem.update()
            self.assertIn(constr1.name, self.model.constraints)
            self.assertIn(constr2.name, self.model.constraints)
            self.assertIn(constr3.name, self.model.constraints)
            self.assertIn(constr4.name, self.model.constraints)
            # constr1
            coeff_dict = dict()
            internal_constraint = self.model.problem.getConstrByName(constr1.name)
            row = self.model.problem.getRow(internal_constraint)
            for i in range(row.size()):
                coeff_dict[row.getVar(i).VarName] = row.getCoeff(i)
            self.assertDictEqual(coeff_dict, {'x': 0.3, 'y': 0.4, 'z': 66., 'test_aux': -1.0})
            self.assertEqual(internal_constraint.RHS, 0)
            self.assertEqual(self.model.problem.getVarByName(internal_constraint.getAttr('ConstrName')+'_aux'), 100)
            # constr2
            coeff_dict = dict()
            internal_constraint = self.model.problem.getConstrByName(constr2.name)
            row = self.model.problem.getRow(internal_constraint)
            for i in range(row.size()):
                coeff_dict[row.getVar(i).VarName] = row.getCoeff(i)
            self.assertDictEqual(coeff_dict, {'x': 2.333, 'y': 1.})
            self.assertEqual(internal_constraint.RHS, 96.997)
            self.assertEqual(internal_constraint.Sense, '<')
            # constr3
            coeff_dict = dict()
            internal_constraint = self.model.problem.getConstrByName(constr3.name)
            print(internal_constraint)
            row = self.model.problem.getRow(internal_constraint)
            for i in range(row.size()):
                coeff_dict[row.getVar(i).VarName] = row.getCoeff(i)
            self.assertDictEqual(coeff_dict, {'x': 2.333, 'y': 1., 'z': 1.})
            self.assertEqual(internal_constraint.RHS, -300)
            self.assertEqual(internal_constraint.Sense, '>')
            # constr4
            coeff_dict = dict()
            internal_constraint = self.model.problem.getConstrByName(constr4.name)
            print(internal_constraint)
            row = self.model.problem.getRow(internal_constraint)
            for i in range(row.size()):
                coeff_dict[row.getVar(i).VarName] = row.getCoeff(i)
            self.assertDictEqual(coeff_dict, {'x': 1})
            self.assertEqual(internal_constraint.RHS, -300)
            self.assertEqual(internal_constraint.Sense, '=')

        def test_remove_constraints(self):
            x = Variable('x', type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=3, ub=3, type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            self.assertEqual(constr1.problem, None)
            self.model.add(constr1)
            self.model.update()
            self.assertEqual(constr1.problem, self.model)
            self.assertIn(constr1.name, self.model.constraints)
            print('test', constr1.name in self.model.constraints.keys())
            self.model.remove(constr1.name)
            self.model.update()
            self.assertEqual(constr1.problem, None)
            self.assertNotIn(constr1, self.model.constraints)

        def test_add_nonlinear_constraint_raises(self):
            x = Variable('x', type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=10, type='integer')
            c = Constraint(0.3 * x + 0.4 * y ** 2 + 66. * z, lb = -100, ub = 0., name = 'test')
            self.model.add(c)
            self.assertRaisesRegexp(ValueError, 'GUROBI currently only supports linear constraint.*', self.model.update)

        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.assertEqual(constraint._internal_constraint, None)
            self.model.add(constraint)
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x')
            z = Variable('z', lb=3, ub=10, type='integer')
            self.assertEqual(z._internal_variable, None)
            constraint += 77. * z
            self.assertEqual(z._internal_variable, self.model.problem.getVarByName('z'))
            self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')

        # def test_change_of_objective_is_reflected_in_low_level_solver(self):
        #     x = Variable('x', lb=-83.3, ub=1324422.)
        #     y = Variable('y', lb=-181133.3, ub=12000.)
        #     objective = Objective(0.3 * x + 0.4 * y, name='test', direction='max')
        #     self.model.objective = objective
        #     self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x')
        #     self.assertEqual(glp_get_obj_coef(self.model.problem, x.index), 0.3)
        #     self.assertEqual(glp_get_obj_coef(self.model.problem, y.index), 0.4)
        #     for i in range(1, glp_get_num_cols(self.model.problem) + 1):
        #         if i != x.index and i != y.index:
        #             self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)
        #     z = Variable('z', lb=4, ub=4, type='integer')
        #     self.model.objective += 77. * z
        #     self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x + 77.0*z')
        #     self.assertEqual(glp_get_obj_coef(self.model.problem, x.index), 0.3)
        #     self.assertEqual(glp_get_obj_coef(self.model.problem, y.index), 0.4)
        #     self.assertEqual(glp_get_obj_coef(self.model.problem, z.index), 77.)
        #     for i in range(1, glp_get_num_cols(self.model.problem) + 1):
        #         if i != x.index and i != y.index and i != z.index:
        #             self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)


        def test_change_variable_bounds(self):
            inner_prob = self.model.problem
            inner_problem_bounds = [(variable.LB, variable.UB) for variable in inner_prob.getVars()]
            bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertEqual(bounds, inner_problem_bounds)
            for var in self.model.variables.values():
                var.lb = random.uniform(-1000, 1000)
                var.ub = random.uniform(var.lb, 1000)
            self.model.update()
            inner_problem_bounds_new = [(variable.LB, variable.UB) for variable in inner_prob.getVars()]
            bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        def test_change_variable_type(self):
            for variable in self.model.variables:
                variable.type = 'integer'
            self.model.update()
            for variable in self.model.problem.getVars():
                self.assertEqual(variable.VType, gurobipy.GRB.INTEGER)

        def test_change_constraint_bounds(self):
            inner_prob = self.model.problem
            inner_problem_bounds = []
            for constr in inner_prob.getConstrs():
                aux_var = inner_prob.getVarByName(constr.getAttr('ConstrName') + '_aux')
                if aux_var is None:
                    inner_problem_bounds.append((constr.RHS, constr.RHS))
                else:
                    inner_problem_bounds.append((aux_var.UB, constr.RHS))
            print(len(self.model.constraints))
            print(len(self.model.problem.getConstrs()))
            bounds = [(constr.lb, constr.ub) for constr in self.model.constraints]
            print('bounds', inner_problem_bounds)
            print('bounds', bounds)
            self.assertEqual(bounds, inner_problem_bounds)
            # for constr in self.model.constraints:
            #     constr.lb = random.uniform(-1000, constr.ub)
            #     constr.ub = random.uniform(constr.lb, 1000)
            # inner_problem_bounds_new = [(glp_get_row_lb(inner_prob, i), glp_get_row_ub(inner_prob, i)) for i in
            #                             range(1, glp_get_num_rows(inner_prob) + 1)]
            # bounds_new = [(constr.lb, constr.ub) for constr in self.model.constraints]
            # self.assertNotEqual(bounds, bounds_new)
            # self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            # self.assertEqual(bounds_new, inner_problem_bounds_new)
            #
        def test_initial_objective(self):
            self.assertEqual(self.model.objective.expression.__str__(), '1.0*R_Biomass_Ecoli_core_w_GAM')

        def test_optimize(self):
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertAlmostEqual(self.model.objective.value, 0.8739215069684303)

        def test_optimize_milp(self):
            problem = gurobipy.read(TESTMILPMODELPATH)
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

        @unittest.skip('Incomplete')
        def test_number_objective(self):
            self.model.objective = Objective(0.)
            self.model.update()
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n0')
            obj_coeff = list()
            print(self.model.problem.getObjective().size())
            print(self.model.problem.getObjective().getVar(0))
            # for i in range(1, glp_get_num_cols(self.model.problem) + 1):
            #     obj_coeff.append(glp_get_obj_coef(self.model.problem, i))
            self.assertEqual(set(obj_coeff), {0.})

        @unittest.skip('Incomplete')
        def test_raise_on_non_linear_objective(self):
            """Test that an exception is raised when a non-linear objective is added to the model."""
            v1, v2 = self.model.variables.values()[0:2]
            self.assertRaises(ValueError, Objective, v1*v2)

        @unittest.skip('Not supported yet')
        def test_iadd_objective(self):
            v2, v3 = self.model.variables.values()[1:3]
            print(v2, v3)
            # 1/0
            self.model.objective += 2. * v2 - 3. * v3
            internal_objective = self.model.problem.getObjective()
            result = {}
            for i in range(internal_objective.size()):
                var = internal_objective.getVar(i)
                coeff = internal_objective.getCoeff(i)
                result[var.VarName] = coeff
            self.assertDictEqual(result, {'R_Biomass_Ecoli_core_w_GAM': 1.0})
            self.model.update()
            self.assertDictEqual(result, {'R_Biomass_Ecoli_core_w_GAM': 1.0, 'R_PGK': 2, 'R_GAPD': -3})

        @unittest.skip('Not supported yet')
        def test_imul_objective(self):
            self.model.objective *= 2.
            obj_coeff = list()
            for i in range(len(self.model.variables)):
                obj_coeff.append(glp_get_obj_coef(self.model.problem, i))
            self.assertEqual(obj_coeff,
                             [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0]
            )

        def test_set_copied_objective(self):
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_Biomass_Ecoli_core_w_GAM')

        def test_timeout(self):
            self.model.configuration.timeout = 0
            status = self.model.optimize()
            print(status)
            self.assertEqual(status, 'time_limit')

        # def test_set_linear_objective_term(self):
        #     self.model._set_linear_objective_term(self.model.variables.R_TPI, 666.)
        #     self.assertEqual(glp_get_obj_coef(self.model.problem, self.model.variables.R_TPI.index), 666.)

        def test_instantiating_model_with_non_glpk_problem_raises(self):
            self.assertRaises(TypeError, Model, problem='Chicken soup')

        # def test__set_coefficients_low_level(self):
        #     constraint = self.model.constraints.M_atp_c
        #     constraint._set_coefficients_low_level({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
        #     num_cols = glp_get_num_cols(self.model.problem)
        #     ia = intArray(num_cols + 1)
        #     da = doubleArray(num_cols + 1)
        #     index = constraint.index
        #     num = glp_get_mat_row(self.model.problem, index, ia, da)
        #     for i in range(1, num +1):
        #         col_name = glp_get_col_name(self.model.problem, ia[i])
        #         if col_name == 'R_Biomass_Ecoli_core_w_GAM':
        #             self.assertEqual(da[i], 666.)

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

        def test_clone_solver(self):
            self.assertEquals(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            cloned_model = Model.clone(self.model)
            self.assertEquals(cloned_model.configuration.verbosity, 3)
            self.assertEquals(len(cloned_model.variables), len(self.model.variables))
            self.assertEquals(len(cloned_model.constraints), len(self.model.constraints))


if __name__ == '__main__':
    nose.runmodule()
