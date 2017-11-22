# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import os
import random
import unittest
import json

import nose

import copy
import pickle

from optlang import glpk_exact_interface
from optlang.tests import abstract_test_cases

from optlang import glpk_interface
from optlang.util import glpk_read_cplex
from swiglpk import glp_get_num_rows, glp_get_col_name, glp_get_num_cols, glp_get_prob_name, glp_get_row_name, \
    glp_get_col_kind, glp_find_col, intArray, doubleArray, glp_get_mat_row, glp_get_row_type, glp_get_row_lb, \
    glp_get_row_ub, glp_get_obj_coef, GLP_UP, GLP_DB, GLP_LO, GLP_CV, GLP_IV, GLP_FX, GLP_FR, glp_get_col_lb, \
    glp_get_col_ub, glp_get_obj_dir, GLP_MIN, GLP_MAX


random.seed(666)
TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')
ECOLI_TEST = os.path.join(os.path.dirname(__file__), 'data/coli_core.json')


class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
    interface = glpk_exact_interface

    def test_get_primal(self):
        self.assertEqual(self.var.primal, None)
        model = glpk_interface.Model(problem=glpk_read_cplex(TESTMODELPATH))
        model.optimize()
        for i, j in zip([var.primal for var in model.variables],
                        [0.8739215069684306, -16.023526143167608, 16.023526143167604, -14.71613956874283,
                         14.71613956874283, 4.959984944574658, 4.959984944574657, 4.959984944574658,
                         3.1162689467973905e-29, 2.926716099010601e-29, 0.0, 0.0, -6.112235045340358e-30,
                         -5.6659435396316186e-30, 0.0, -4.922925402711085e-29, 0.0, 9.282532599166613, 0.0,
                         6.00724957535033, 6.007249575350331, 6.00724957535033, -5.064375661482091, 1.7581774441067828,
                         0.0, 7.477381962160285, 0.0, 0.22346172933182767, 45.514009774517454, 8.39, 0.0,
                         6.007249575350331, 0.0, -4.541857463865631, 0.0, 5.064375661482091, 0.0, 0.0,
                         2.504309470368734, 0.0, 0.0, -22.809833310204958, 22.809833310204958, 7.477381962160285,
                         7.477381962160285, 1.1814980932459636, 1.496983757261567, -0.0, 0.0, 4.860861146496815, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 5.064375661482091, 0.0, 5.064375661482091, 0.0, 0.0,
                         1.496983757261567, 10.000000000000002, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -29.175827135565804,
                         43.598985311997524, 29.175827135565804, 0.0, 0.0, 0.0, -1.2332237321082153e-29,
                         3.2148950476847613, 38.53460965051542, 5.064375661482091, 0.0, -1.2812714099825612e-29,
                         -1.1331887079263237e-29, 17.530865429786694, 0.0, 0.0, 0.0, 4.765319193197458,
                         -4.765319193197457, 21.79949265599876, -21.79949265599876, -3.2148950476847613, 0.0,
                         -2.281503094067127, 2.6784818505075303, 0.0]):
            self.assertAlmostEqual(i, j)

    def test_changing_variable_names_is_reflected_in_the_solver(self):
        model = self.interface.Model(problem=glpk_read_cplex(TESTMODELPATH))
        for i, variable in enumerate(model.variables):
            variable.name = "var" + str(i)
            self.assertEqual(variable.name, "var" + str(i))
            self.assertEqual(glp_get_col_name(model.problem, variable._index), "var" + str(i))

    def test_change_type(self):
        self.var.type = "continuous"
        self.assertRaises(ValueError, setattr, self.var, "type", "integer")
        self.assertRaises(ValueError, setattr, self.var, "type", "binary")

    def test_set_wrong_type_raises(self):
        self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
        self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
        self.model.add(self.var)
        self.model.update()
        self.assertRaises(ValueError, setattr, self.var, "type", "mustard")


class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
    interface = glpk_exact_interface

    def test_get_primal(self):
        pass

    @unittest.skip("NA")
    def test_indicator_constraint_support(self):
        pass


class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
    interface = glpk_exact_interface

    def setUp(self):
        with open(ECOLI_TEST) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))
        self.obj = self.model.objective

    def test_change_direction(self):
        self.obj.direction = "min"
        self.assertEqual(self.obj.direction, "min")
        self.assertEqual(glp_get_obj_dir(self.model.problem), GLP_MIN)

        self.obj.direction = "max"
        self.assertEqual(self.obj.direction, "max")
        self.assertEqual(glp_get_obj_dir(self.model.problem), GLP_MAX)


class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):
    interface = glpk_exact_interface


class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
    interface = glpk_exact_interface

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
        inner_prob = self.model.problem
        self.assertEqual(len(self.model.variables), glp_get_num_cols(inner_prob))
        self.assertEqual(len(self.model.constraints), glp_get_num_rows(inner_prob))
        self.assertEqual(self.model.variables.keys(),
                         [glp_get_col_name(inner_prob, i) for i in range(1, glp_get_num_cols(inner_prob) + 1)])
        self.assertEqual(self.model.constraints.keys(),
                         [glp_get_row_name(inner_prob, j) for j in range(1, glp_get_num_rows(inner_prob) + 1)])

    def test_add_non_cplex_conform_variable(self):
        var = self.interface.Variable('12x!!@#5_3', lb=-666, ub=666)
        self.assertEqual(var._index, None)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(var.name, glp_get_col_name(self.model.problem, var._index))
        self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
        self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
        repickled = pickle.loads(pickle.dumps(self.model))
        var_from_pickle = repickled.variables['12x!!@#5_3']
        self.assertEqual(var_from_pickle.name, glp_get_col_name(repickled.problem, var_from_pickle._index))

    def test_change_of_constraint_is_reflected_in_low_level_solver(self):
        x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        constraint = self.interface.Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
        self.assertEqual(constraint._index, None)
        self.model.add(constraint)
        self.assertEqual(
            (self.model.constraints["test"].expression - (0.4 * y + 0.3 * x)).expand(), 0
        )
        self.assertEqual(self.model.constraints["test"].lb, -100)

        self.assertEqual(constraint._index, 73)
        z = self.interface.Variable('z', lb=3, ub=10, type='continuous')
        self.assertEqual(z._index, None)
        constraint += 77. * z
        self.assertEqual(z._index, 98)

        self.assertEqual(
            (self.model.constraints["test"].expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand(), 0
        )
        self.assertEqual(self.model.constraints["test"].lb, -100)

        self.assertEqual(constraint._index, 73)

    def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
        x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        constraint = self.interface.Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
        self.model.add(constraint)
        z = self.interface.Variable('z', lb=2, ub=5, type='continuous')
        constraint += 77. * z
        self.model.remove(constraint)
        self.assertEqual(
            (constraint.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0, 0
        )
        self.assertEqual(constraint.lb, -100)
        self.assertEqual(constraint.ub, None)

    def test_change_of_objective_is_reflected_in_low_level_solver(self):
        x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        objective = self.interface.Objective(0.3 * x + 0.4 * y, name='test', direction='max')
        self.model.objective = objective

        self.assertEqual(
            (self.model.objective.expression - (0.4 * y + 0.3 * x)).expand() - 0, 0
        )
        self.assertEqual(self.model.objective.direction, "max")

        self.assertEqual(glp_get_obj_coef(self.model.problem, x._index), 0.3)
        self.assertEqual(glp_get_obj_coef(self.model.problem, y._index), 0.4)
        for i in range(1, glp_get_num_cols(self.model.problem) + 1):
            if i != x._index and i != y._index:
                self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)
        z = self.interface.Variable('z', lb=4, ub=4, type='continuous')
        self.model.objective += 77. * z

        self.assertEqual(
            (self.model.objective.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0, 0
        )
        self.assertEqual(self.model.objective.direction, "max")

        self.assertEqual(glp_get_obj_coef(self.model.problem, x._index), 0.3)
        self.assertEqual(glp_get_obj_coef(self.model.problem, y._index), 0.4)
        self.assertEqual(glp_get_obj_coef(self.model.problem, z._index), 77.)
        for i in range(1, glp_get_num_cols(self.model.problem) + 1):
            if i != x._index and i != y._index and i != z._index:
                self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)

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

    def test_iadd_objective(self):
        v2, v3 = self.model.variables.values()[1:3]
        self.model.objective += 2. * v2 - 3. * v3
        obj_coeff = list()
        for i in range(len(self.model.variables)):
            obj_coeff.append(glp_get_obj_coef(self.model.problem, i))
        self.assertEqual(obj_coeff,
                         [0.0, 1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0]
                         )

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
        self.assertEqual(
            (self.model.objective.expression - 1.0 * self.model.variables.R_Biomass_Ecoli_core_w_GAM).expand() - 0, 0
        )
        self.assertEqual(self.model.objective.direction, "max")

    def test_timeout(self):
        self.model.configuration.timeout = 0
        status = self.model.optimize()
        self.assertEqual(status, 'time_limit')

    def test_set_linear_coefficients_objective(self):
        self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
        self.assertEqual(glp_get_obj_coef(self.model.problem, self.model.variables.R_TPI._index), 666.)

    def test_set_linear_coefficients_constraint(self):
        constraint = self.model.constraints.M_atp_c
        constraint.set_linear_coefficients({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
        num_cols = glp_get_num_cols(self.model.problem)
        ia = intArray(num_cols + 1)
        da = doubleArray(num_cols + 1)
        index = constraint._index
        num = glp_get_mat_row(self.model.problem, index, ia, da)
        for i in range(1, num + 1):
            col_name = glp_get_col_name(self.model.problem, ia[i])
            if col_name == 'R_Biomass_Ecoli_core_w_GAM':
                self.assertEqual(da[i], 666.)

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

    def test_add_nonlinear_constraint_raises(self):
        x = self.interface.Variable('x', type='continuous')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=3, ub=3, type='continuous')
        with self.assertRaises(ValueError):
            constraint = self.interface.Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
            self.model.add(constraint)
            self.model.update()

    def test_change_variable_type(self):
        self.assertRaises(ValueError, setattr, self.model.variables[-1], "type", "integer")

    def test_add_integer_var(self):
        self.assertRaises(ValueError, self.interface.Variable, 'int_var', lb=-13, ub=499., type='integer')

    def test_is_integer(self):
        self.skipTest("No integers with glpk_exact")

    def test_binary_variables(self):
        self.skipTest("No integers with glpk_exact")

    def test_implicitly_convert_milp_to_lp(self):
        self.skipTest("No integers with glpk_exact")

    def test_optimize_milp(self):
        self.skipTest("No integers with glpk_exact")

    def test_integer_variable_dual(self):
        self.skipTest("No integers with glpk_exact")

    def test_integer_constraint_dual(self):
        self.skipTest("No integers with glpk_exact")

    def test_integer_batch_duals(self):
        self.skipTest("No integers with glpk_exact")


if __name__ == '__main__':
    nose.runmodule()
