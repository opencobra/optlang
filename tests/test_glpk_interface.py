# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.
import copy

import unittest
import random
import pickle

import os
import nose
import re
from swiglpk import *

from optlang.glpk_interface import Variable, Constraint, Model, Objective


random.seed(666)
TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')


class SolverTestCase(unittest.TestCase):
    def setUp(self):
        glp_term_out(GLP_OFF)
        problem = glp_create_prob()
        glp_read_lp(problem, None, TESTMODELPATH)
        assert glp_get_num_cols(problem) > 0
        self.model = Model(problem=problem)

    def test_create_empty_model(self):
        model = Model()
        self.assertEqual(glp_get_num_cols(model.problem), 0)
        self.assertEqual(glp_get_num_rows(model.problem), 0)
        self.assertEqual(model.name, None)
        self.assertEqual(glp_get_prob_name(model.problem), None)
        model = Model(name="empty_problem")
        self.assertEqual(glp_get_prob_name(model.problem), "empty_problem")

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

    def test_copy(self):
        model_copy = copy.copy(self.model)
        self.assertNotEqual(id(self.model), id(model_copy))
        self.assertNotEqual(id(self.model.problem), id(model_copy.problem))

    def test_init_from_existing_problem(self):
        inner_prob = self.model.problem
        self.assertEqual(len(self.model.variables), glp_get_num_cols(inner_prob))
        self.assertEqual(len(self.model.constraints), glp_get_num_rows(inner_prob))
        self.assertEqual(self.model.variables.keys(),
                         [glp_get_col_name(inner_prob, i) for i in range(1, glp_get_num_cols(inner_prob) + 1)])
        self.assertEqual(self.model.constraints.keys(),
                         [glp_get_row_name(inner_prob, j) for j in range(1, glp_get_num_rows(inner_prob) + 1)])

    def test_add_variable(self):
        var = Variable('x')
        self.assertEqual(var.index, None)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables.values().count(var), 1)
        self.assertEqual(var.index, glp_get_num_cols(self.model.problem))
        self.assertEqual(var.name, glp_get_col_name(self.model.problem, var.index))
        self.assertEqual(self.model.variables['x'].problem, var.problem)
        self.assertEqual(glp_get_col_kind(self.model.problem, var.index), GLP_CV)
        var = Variable('y', lb=-13)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(var.name, glp_get_col_name(self.model.problem, var.index))
        self.assertEqual(glp_get_col_kind(self.model.problem, var.index), GLP_CV)
        self.assertEqual(self.model.variables['x'].lb, None)
        self.assertEqual(self.model.variables['x'].ub, None)
        self.assertEqual(self.model.variables['y'].lb, -13)
        self.assertEqual(self.model.variables['x'].ub, None)
        var = Variable('x_with_ridiculously_long_variable_name_asdffffffffasdfasdfasdfasdfasdfasdfasdf')
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables.values().count(var), 1)
        var = Variable('x_with_ridiculously_long_variable_name_asdffffffffasdfasdfasdfasdfasdfasdfasdf')
        self.assertRaises(Exception, self.model.add, var)
        self.assertEqual(len(self.model.variables), glp_get_num_cols(self.model.problem))

    def test_add_integer_var(self):
        var = Variable('int_var', lb=-13, ub=499.4, type='integer')
        self.model.add(var)
        self.assertEqual(self.model.variables['int_var'].type, 'integer')
        self.assertEqual(glp_get_col_kind(self.model.problem, var.index), GLP_IV)
        self.assertEqual(self.model.variables['int_var'].ub, 499.4)
        self.assertEqual(self.model.variables['int_var'].lb, -13)

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

    def test_remove_variable(self):
        var = self.model.variables.values()[0]
        self.assertEqual(self.model.constraints['M_atp_c'].__str__(),
                         'M_atp_c: -1.0*R_ACKr - 1.0*R_ADK1 + 1.0*R_ATPS4r - 1.0*R_PGK - 1.0*R_SUCOAS - 59.81*R_Biomass_Ecoli_core_w_GAM - 1.0*R_GLNS - 1.0*R_GLNabc - 1.0*R_PFK - 1.0*R_PPCK - 1.0*R_PPS + 1.0*R_PYK - 1.0*R_ATPM')
        self.assertEqual(var.problem, self.model)
        self.model.remove(var)
        self.assertEqual(self.model.constraints['M_atp_c'].__str__(),
                         'M_atp_c: -1.0*R_ACKr - 1.0*R_ADK1 + 1.0*R_ATPS4r - 1.0*R_PGK - 1.0*R_SUCOAS - 1.0*R_GLNS - 1.0*R_GLNabc - 1.0*R_PFK - 1.0*R_PPCK - 1.0*R_PPS + 1.0*R_PYK - 1.0*R_ATPM')
        self.assertNotIn(var, self.model.variables.values())
        self.assertEqual(glp_find_col(self.model.problem, var.name), 0)
        self.assertEqual(var.problem, None)

    def test_remove_variable_str(self):
        var = self.model.variables.values()[0]
        self.model.remove(var.name)
        self.assertNotIn(var, self.model.variables.values())
        self.assertEqual(glp_find_col(self.model.problem, var.name), 0)
        self.assertEqual(var.problem, None)

    def test_add_constraints(self):
        x = Variable('x', lb=-83.3, ub=1324422., type='binary')
        y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
        constr3 = Constraint(2.333 * x + y + z, lb=-300)
        self.model.add(constr1)
        self.model.add(constr2)
        self.model.add(constr3)
        self.assertIn(constr1.name, self.model.constraints)
        self.assertIn(constr2.name, self.model.constraints)
        self.assertIn(constr3.name, self.model.constraints)
        # constr1
        ia = intArray(glp_get_num_rows(self.model.problem) + 1)
        da = doubleArray(glp_get_num_rows(self.model.problem) + 1)
        nnz = glp_get_mat_row(self.model.problem, constr1.index, ia, da)
        coeff_dict = dict()
        for i in range(1, nnz+1):
            coeff_dict[glp_get_col_name(self.model.problem, ia[i])] = da[i]
        self.assertDictEqual(coeff_dict, {'x': 0.3, 'y': 0.4, 'z': 66.})
        self.assertEqual(glp_get_row_type(self.model.problem, constr1.index), GLP_DB)
        self.assertEqual(glp_get_row_lb(self.model.problem, constr1.index), -100)
        self.assertEqual(glp_get_row_ub(self.model.problem, constr1.index), 0)
        # constr2
        ia = intArray(glp_get_num_rows(self.model.problem) + 1)
        da = doubleArray(glp_get_num_rows(self.model.problem) + 1)
        nnz = glp_get_mat_row(self.model.problem, constr2.index, ia, da)
        coeff_dict = dict()
        for i in range(1, nnz+1):
            coeff_dict[glp_get_col_name(self.model.problem, ia[i])] = da[i]
        self.assertDictEqual(coeff_dict, {'x': 2.333, 'y': 1.})
        self.assertEqual(glp_get_row_type(self.model.problem, constr2.index), GLP_UP)
        self.assertEqual(glp_get_row_lb(self.model.problem, constr2.index), -1.7976931348623157e+308)
        self.assertEqual(glp_get_row_ub(self.model.problem, constr2.index), 96.997)
        # constr3
        ia = intArray(glp_get_num_rows(self.model.problem) + 1)
        da = doubleArray(glp_get_num_rows(self.model.problem) + 1)
        nnz = glp_get_mat_row(self.model.problem, constr3.index, ia, da)
        coeff_dict = dict()
        for i in range(1, nnz+1):
            coeff_dict[glp_get_col_name(self.model.problem, ia[i])] = da[i]
        self.assertDictEqual(coeff_dict, {'x': 2.333, 'y': 1., 'z': 1.})
        self.assertEqual(glp_get_row_type(self.model.problem, constr3.index), GLP_LO)
        self.assertEqual(glp_get_row_lb(self.model.problem, constr3.index), -300)
        self.assertEqual(glp_get_row_ub(self.model.problem, constr3.index), 1.7976931348623157e+308)

    def test_remove_constraints(self):
        x = Variable('x', lb=-83.3, ub=1324422., type='binary')
        y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        self.assertEqual(constr1.problem, None)
        self.model.add(constr1)
        self.assertEqual(constr1.problem, self.model)
        self.assertIn(constr1.name, self.model.constraints)
        print constr1.index
        self.model.remove(constr1.name)
        self.assertEqual(constr1.problem, None)
        self.assertNotIn(constr1, self.model.constraints)

    def test_add_nonlinear_constraint_raises(self):
        x = Variable('x', lb=-83.3, ub=1324422., type='binary')
        y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        self.assertRaises(ValueError, Constraint, 0.3 * x + 0.4 * y ** 2 + 66. * z, lb=-100, ub=0., name='test')

    def test_change_of_constraint_is_reflected_in_low_level_solver(self):
        x = Variable('x', lb=-83.3, ub=1324422.)
        y = Variable('y', lb=-181133.3, ub=12000.)
        constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
        self.assertEqual(constraint.index, None)
        self.model.add(constraint)
        self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x')
        self.assertEqual(constraint.index, 73)
        z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        self.assertEqual(z.index, None)
        constraint += 77. * z
        self.assertEqual(z.index, 98)
        self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x + 77.0*z')
        print self.model
        self.assertEqual(constraint.index, 73)

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
        z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
        self.model.objective += 77. * z
        self.assertEqual(self.model.objective.__str__(), 'Maximize\n0.4*y + 0.3*x + 77.0*z')
        self.assertEqual(glp_get_obj_coef(self.model.problem, x.index), 0.3)
        self.assertEqual(glp_get_obj_coef(self.model.problem, y.index), 0.4)
        self.assertEqual(glp_get_obj_coef(self.model.problem, z.index), 77.)
        for i in range(1, glp_get_num_cols(self.model.problem) + 1):
            if i != x.index and i != y.index and i != z.index:
                self.assertEqual(glp_get_obj_coef(self.model.problem, i), 0)

    @unittest.skip('Skipping for now')
    def test_absolute_value_objective(self):
        # TODO: implement hack mentioned in http://www.aimms.com/aimms/download/manuals/aimms3om_linearprogrammingtricks.pdf

        objective = Objective(sum(abs(variable) for variable in self.model.variables.itervalues()), name='test',
                              direction='max')
        print objective
        self.assertTrue(False)

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

    def test_change_variable_type(self):
        for variable in self.model.variables:
            variable.type = 'integer'
        for i in range(1, glp_get_num_cols(self.model.problem) + 1):
            self.assertEqual(glp_get_col_kind(self.model.problem, i), GLP_IV)

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
        self.assertEqual(self.model.objective.expression.__str__(), '1.0*R_Biomass_Ecoli_core_w_GAM')

    def test_optimize(self):
        self.model.optimize()
        self.assertEqual(self.model.status, 'optimal')
        self.assertAlmostEqual(self.model.objective.value, 0.8739215069684303)

    def test_change_objective(self):
        """Test that all different kinds of linear objective specification work."""
        print self.model.variables.values()[0:2]
        v1, v2 = self.model.variables.values()[0:2]
        self.model.objective = Objective(1. * v1 + 1. * v2)
        self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_PGK + 1.0*R_Biomass_Ecoli_core_w_GAM')
        self.model.objective = Objective(v1 + v2)
        self.assertEqual(self.model.objective.__str__(), 'Maximize\n1.0*R_PGK + 1.0*R_Biomass_Ecoli_core_w_GAM')

    def test_number_objective(self):
        self.model.objective = Objective(0.)
        self.assertEqual(self.model.objective.__str__(), 'Maximize\n0')
        obj_coeff = list()
        for i in xrange(1, glp_get_num_cols(self.model.problem) + 1):
            obj_coeff.append(glp_get_obj_coef(self.model.problem, i))
        self.assertEqual(set(obj_coeff), {0.})

    def test_raise_on_non_linear_objective(self):
        """Test that an exception is raised when a non-linear objective is added to the model."""
        v1, v2 = self.model.variables.values()[0:2]
        self.assertRaises(ValueError, Objective, v1*v2)

    def test_iadd_objective(self):
        v2, v3 = self.model.variables.values()[1:3]
        self.model.objective += 2. * v2 - 3. * v3
        obj_coeff = list()
        for i in xrange(len(self.model.variables)):
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
        for i in xrange(len(self.model.variables)):
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
        self.assertEqual(status, 'infeasible')

if __name__ == '__main__':
    nose.runmodule()
