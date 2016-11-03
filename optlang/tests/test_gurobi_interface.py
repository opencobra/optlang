# Copyright (c) 2016 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import unittest

try:  # noqa: C901
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
    from optlang.tests import abstract_test_cases
    from optlang import gurobi_interface

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
    TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')


    class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
        interface = gurobi_interface

        def test_internal_variable(self):
            self.assertEqual(self.var._internal_variable, None)

        def test_gurobi_change_name(self):
            self.model.add(self.var)
            self.model.update()
            self.var.name = "test_2"
            self.assertEqual(self.var._internal_variable.getAttr("VarName"), "test_2")

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            model.optimize()
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

        def test_changing_variable_names_is_reflected_in_the_solver(self):
            model = Model(problem=gurobipy.read(TESTMODELPATH))
            for i, variable in enumerate(model.variables):
                print(variable._internal_variable is not None)
                print(variable.problem.name)
                variable.name = "var" + str(i)
                print(variable.problem.name)
                print(variable.name)
                print(variable._internal_variable is not None)
                self.assertEqual(variable.name, "var" + str(i))
                self.assertEqual(variable._internal_variable.getAttr('VarName'), "var" + str(i))

        def test_gurobi_setting_bounds(self):
            var = self.var
            model = self.model
            model.add(var)
            model.update()
            var.lb = 1
            self.assertEqual(var.lb, 1)
            model.problem.update()
            self.assertEqual(var._internal_variable.getAttr('LB'), 1)
            var.ub = 2
            self.assertEqual(var.ub, 2)
            model.problem.update()
            self.assertEqual(var._internal_variable.getAttr('UB'), 2)


    class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
        interface = gurobi_interface

        def test_get_primal(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            print([constraint.primal for constraint in self.model.constraints])
            for i, j in zip([constraint.primal for constraint in self.model.constraints],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             4.048900234729145e-15, 0.0, 0.0, 0.0, -3.55971196577979e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 2.5546369406238147e-17, 0.0, -5.080374405378186e-29, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                self.assertAlmostEqual(i, j)


    class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
        def setUp(self):
            problem = gurobipy.read(TESTMODELPATH)
            self.model = Model(problem=problem)
            self.obj = self.model.objective

        def test_change_direction(self):
            self.obj.direction = "min"
            self.assertEqual(self.obj.direction, "min")
            self.assertEqual(self.model.problem.getAttr('ModelSense'), gurobipy.GRB.MAXIMIZE)
            self.model.update()
            self.assertEqual(self.model.problem.getAttr('ModelSense'), gurobipy.GRB.MINIMIZE)

            self.obj.direction = "max"
            self.assertEqual(self.obj.direction, "max")
            self.assertEqual(self.model.problem.getAttr('ModelSense'), gurobipy.GRB.MINIMIZE)
            self.model.update()
            self.assertEqual(self.model.problem.getAttr('ModelSense'), gurobipy.GRB.MAXIMIZE)


    class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
        interface = gurobi_interface

        def test_gurobi_create_empty_model(self):
            model = Model()
            self.assertEqual(model.problem.getAttr('NumVars'), 0)
            self.assertEqual(model.problem.getAttr('NumConstrs'), 0)
            self.assertEqual(model.name, None)
            self.assertEqual(model.problem.getAttr('ModelName'), '')
            model = Model(name="empty_problem")
            self.assertEqual(model.problem.getAttr('ModelName'), 'empty_problem')

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

        def test_init_from_existing_problem(self):
            self.assertEqual(len(self.model.variables), len(self.model.problem.getVars()))
            self.assertEqual(len(self.model.constraints), len(self.model.problem.getConstrs()))
            self.assertEqual(self.model.variables.keys(),
                             [var.VarName for var in self.model.problem.getVars()])

            self.assertEqual(self.model.constraints.keys(),
                             [constr.ConstrName for constr in self.model.problem.getConstrs()])

        def test_gurobi_add_variable(self):
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

        def test_gurobi_add_integer_var(self):
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

        def test_gurobi_add_constraints(self):
            x = Variable('x', lb=0, ub=1, type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0., ub=10., type='integer')
            constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y + z, lb=-300)
            constr4 = Constraint(x, lb=-300, ub=-300)
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
            self.assertEqual(self.model.problem.getVarByName(internal_constraint.getAttr('ConstrName') + '_aux'), 100)
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
            objective = Objective(0.3 * x + 0.4 * y, name='test', direction='max')
            self.model.objective = objective
            self.model.update()
            grb_obj = self.model.problem.getObjective()
            grb_x = self.model.problem.getVarByName(x.name)
            grb_y = self.model.problem.getVarByName(y.name)
            expected = {grb_x: 0.3, grb_y: 0.4}
            for i in range(grb_obj.size()):
                self.assertEqual(grb_obj.getCoeff(i), expected[grb_obj.getVar(i)])
            z = Variable('z', lb=4, ub=4, type='integer')
            grb_z = self.model.problem.getVarByName(z.name)
            self.model.objective += 77. * z
            expected[grb_z] = 77.
            self.model.update()
            for i in range(grb_obj.size()):
                self.assertEqual(grb_obj.getCoeff(i), expected[grb_obj.getVar(i)])

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

        def test_gurobi_change_variable_type(self):
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

        def test_initial_objective(self):
            self.assertEqual(self.model.objective.expression.__str__(), '1.0*R_Biomass_Ecoli_core_w_GAM')

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

        def test_set_copied_objective(self):
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_Biomass_Ecoli_core_w_GAM')

        def test_timeout(self):
            self.model.configuration.timeout = 0
            status = self.model.optimize()
            print(status)
            self.assertEqual(status, 'time_limit')

        def test_set_linear_coefficients_objective(self):
            self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
            self.model.update()
            grb_obj = self.model.problem.getObjective()
            for i in range(grb_obj.size()):
                if 'R_TPI' == grb_obj.getVar(i).getAttr('VarName'):
                    self.assertEqual(grb_obj.getCoeff(i), 666.)

        def test_set_linear_coefficients_constraint(self):
            constraint = self.model.constraints.M_atp_c
            constraint.set_linear_coefficients({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
            self.model.update()
            row = self.model.problem.getRow(self.model.problem.getConstrByName(constraint.name))
            for i in range(row.size()):
                col_name = row.getVar(i).getAttr('VarName')
                if col_name == 'R_Biomass_Ecoli_core_w_GAM':
                    self.assertEqual(row.getCoeff(i), 666.)

if __name__ == '__main__':
    nose.runmodule()
