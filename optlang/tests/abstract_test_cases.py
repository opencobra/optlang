# Copyright 2016 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import abc
import unittest

import six
from optlang import interface
import optlang
import pickle
import json
import copy
import os
import sympy

__test__ = False

TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.json')
TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.json')


@six.add_metaclass(abc.ABCMeta)
class AbstractVariableTestCase(unittest.TestCase):
    @unittest.skip('Abstract test method not implemented.')
    def test_magic(self):
        pass

    def setUp(self):
        self.var = self.interface.Variable('test')
        self.model = self.interface.Model()

    def test_set_wrong_type_raises(self):
        self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
        self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
        self.model.add(self.var)
        self.model.update()
        self.assertRaises(ValueError, setattr, self.var, "type", "mustard")
        self.var.type = "integer"
        self.assertEqual(self.var.type, "integer")

    def test_change_type(self):
        var = self.interface.Variable("test")
        var.type = "binary"
        self.assertEqual(var.type, "binary")

    def test_change_name(self):
        self.model.add(self.var)
        self.model.update()
        self.var.name = "test_2"
        self.assertEqual(self.var.name, "test_2")

    @abc.abstractmethod
    def test_get_primal(self):
        pass

    def test_get_dual(self):
        with open(TESTMODELPATH) as infile:
            model = self.interface.Model.from_json(json.load(infile))
        model.optimize()
        self.assertEqual(model.status, 'optimal')
        self.assertAlmostEqual(model.objective.value, 0.8739215069684305)
        self.assertTrue(isinstance(model.variables[0].dual, float))

    def test_setting_lower_bound_higher_than_upper_bound_raises(self):
        self.model.add(self.var)
        self.var.ub = 0
        self.assertRaises(ValueError, setattr, self.model.variables[0], 'lb', 100.)

    def test_setting_nonnumerical_bounds_raises(self):
        self.assertRaises(TypeError, setattr, self.var, "lb", "Minestrone")
        self.assertRaises(TypeError, setattr, self.var, "ub", "Minestrone")
        self.model.add(self.var)
        self.assertRaises(TypeError, setattr, self.model.variables[0], 'lb', 'Chicken soup')
        self.assertRaises(TypeError, setattr, self.model.variables[0], 'ub', 'Chicken soup')

    @abc.abstractmethod
    def test_changing_variable_names_is_reflected_in_the_solver(self):
        pass

    def test_setting_bounds(self):
        self.var.ub = 5
        self.model.objective = self.interface.Objective(self.var)
        self.model.optimize()
        self.assertEqual(self.var.primal, 5)
        self.var.ub = 4
        self.model.optimize()
        self.assertEqual(self.var.primal, 4)
        self.var.lb = -3
        self.model.objective.direction = "min"
        self.model.optimize()
        self.assertEqual(self.var.primal, -3)
        self.var.lb = sympy.Number(-4)  # Sympy numbers should be valid bounds
        self.model.optimize()
        self.assertEqual(self.var.primal, -4)

    def test_set_bounds_method(self):
        var = self.interface.Variable("test", lb=-10)
        c = self.interface.Constraint(var, lb=-100)
        model = self.interface.Model()
        obj = self.interface.Objective(var)
        model.add(c)
        model.objective = obj

        for lb, ub in ((1, 10), (-1, 5), (11, 12)):
            obj.direction = "max"
            var.set_bounds(lb, ub)
            model.optimize()
            self.assertAlmostEqual(var.primal, ub)
            obj.direction = "min"
            model.optimize()
            self.assertAlmostEqual(var.primal, lb)

        var.set_bounds(None, 0)
        model.optimize()
        self.assertAlmostEqual(var.primal, -100)

        obj.direction = "max"
        var.set_bounds(1, None)
        self.assertEqual(model.optimize(), optlang.interface.UNBOUNDED)

        self.assertRaises(ValueError, var.set_bounds, 2, 1)

    def test_set_bounds_to_none(self):
        model = self.model
        var = self.var
        model.objective = self.interface.Objective(var)
        self.assertEqual(model.optimize(), interface.UNBOUNDED)
        var.ub = 10
        self.assertEqual(model.optimize(), interface.OPTIMAL)
        var.ub = None
        self.assertEqual(model.optimize(), interface.UNBOUNDED)
        self.model.objective.direction = "min"
        var.lb = -10
        self.assertEqual(model.optimize(), interface.OPTIMAL)
        var.lb = None
        self.assertEqual(model.optimize(), interface.UNBOUNDED)


@six.add_metaclass(abc.ABCMeta)
class AbstractConstraintTestCase(unittest.TestCase):

    def setUp(self):
        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))
        self.constraint = self.interface.Constraint(
            self.interface.Variable('chip') + self.interface.Variable('chap'),
            name='woodchips',
            lb=100
        )

    def test_indicator_constraint_support(self):
        if self.interface.Constraint._INDICATOR_CONSTRAINT_SUPPORT:
            constraint = self.interface.Constraint(
                self.interface.Variable('chip_2'),
                indicator_variable=self.interface.Variable('chip', type='binary'), active_when=0, lb=0,
                ub=0,
                name='indicator_constraint_fwd_1'
            )
            model = self.interface.Model()
            model.add(constraint)
            model.update()
        else:
            self.assertRaises(
                optlang.exceptions.IndicatorConstraintsNotSupported,
                self.interface.Constraint,
                self.interface.Variable('chip') + self.interface.Variable('chap'),
                indicator_variable=self.interface.Variable('indicator', type='binary')
            )

    @abc.abstractmethod
    def test_get_primal(self):
        pass

    def test_get_dual(self):
        self.assertEqual(self.constraint.dual, None)
        self.model.optimize()
        self.assertEqual(self.model.status, 'optimal')
        self.assertAlmostEqual(self.model.objective.value, 0.8739215069684305)
        self.assertTrue(isinstance(self.model.constraints[0].dual, float))

    def test_change_constraint_name(self):
        constraint = self.interface.Constraint.clone(self.constraint)
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
        self.assertRaises(ValueError, setattr, self.model.constraints[0], 'lb', 10000000000.)
        self.assertRaises(ValueError, setattr, self.model.constraints[0], "ub", -1000000000.)

        self.assertRaises(ValueError, self.interface.Constraint, 0, lb=0, ub=-1)

    def test_setting_bounds(self):
        var = self.interface.Variable("test", lb=-10)
        c = self.interface.Constraint(var, lb=0)
        model = self.interface.Model()
        obj = self.interface.Objective(var)
        model.add(c)
        model.objective = obj

        c.ub = 5
        model.optimize()
        self.assertEqual(var.primal, 5)
        c.ub = 4
        model.optimize()
        self.assertEqual(var.primal, 4)
        c.lb = -3
        model.objective.direction = "min"
        model.optimize()
        self.assertEqual(var.primal, -3)
        c.lb = sympy.Number(-4)  # Sympy numbers should be valid bounds
        model.optimize()
        self.assertEqual(var.primal, -4)

    def test_setting_nonnumerical_bounds_raises(self):
        var = self.interface.Variable("test")
        constraint = self.interface.Constraint(var, lb=0)
        self.assertRaises(TypeError, setattr, constraint, "lb", "noodle soup")
        self.assertRaises(TypeError, setattr, self.model.constraints[0], 'lb', 'Chicken soup')
        self.assertRaises(TypeError, setattr, constraint, "ub", "noodle soup")
        self.assertRaises(TypeError, setattr, self.model.constraints[0], 'ub', 'Chicken soup')

    def test_set_constraint_bounds_to_none(self):
        model = self.interface.Model()
        var = self.interface.Variable("test")
        const = self.interface.Constraint(var, lb=-10, ub=10)
        obj = self.interface.Objective(var)
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


@six.add_metaclass(abc.ABCMeta)
class AbstractObjectiveTestCase(unittest.TestCase):
    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def test_change_direction(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class AbstractModelTestCase(unittest.TestCase):

    def setUp(self):
        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))

    def test_create_empty_model(self):
        model = self.interface.Model()
        self.assertEqual(len(model.constraints), 0)
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(model.objective.expression, 0)

    @abc.abstractmethod
    def test_pickle_ability(self):
        pass

    def test_pickle_empty_model(self):
        model = self.interface.Model()
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
        self.assertIsNot(self.model, model_copy)
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
        self.assertIsNot(self.model, model_copy)
        self.assertIsNot(self.model.problem, model_copy.problem)
        model_copy.optimize()
        self.assertAlmostEqual(value, model_copy.objective.value)
        self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in model_copy.variables.values()],
                         [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
        self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in model_copy.constraints],
                         [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

    @abc.abstractmethod
    def test_config_gets_copied_too(self):
        pass

    @abc.abstractmethod
    def test_init_from_existing_problem(self):
        pass

    def test_add_variable(self):
        model = self.interface.Model()
        var = self.interface.Variable('x')
        self.assertEqual(var.problem, None)
        model.add(var)
        self.assertTrue(var in model.variables)
        self.assertEqual(model.variables['x'].problem, var.problem)
        self.assertEqual(model.variables['x'].problem, model)
        var = self.interface.Variable('y', lb=-13)
        model.add(var)
        self.assertTrue(var in model.variables)
        self.assertEqual(model.variables['x'].lb, None)
        self.assertEqual(model.variables['x'].ub, None)
        self.assertEqual(model.variables['y'].lb, -13)
        self.assertEqual(model.variables['x'].ub, None)

    def test_add_integer_var(self):
        var = self.interface.Variable('int_var', lb=-13, ub=499., type='integer')
        self.model.add(var)
        self.assertEqual(self.model.variables['int_var'].type, 'integer')
        self.assertEqual(self.model.variables['int_var'].ub, 499.)
        self.assertEqual(self.model.variables['int_var'].lb, -13)

    @abc.abstractmethod
    def test_add_non_cplex_conform_variable(self):
        pass

    def test_remove_variable(self):
        var = self.model.variables[0]
        self.assertEqual(var.problem, self.model)
        self.model.remove(var)
        self.assertNotIn(var, self.model.variables)
        self.assertEqual(var.problem, None)

    def test_remove_variable_str(self):
        var = self.model.variables.values()[0]
        self.model.remove(var.name)
        self.assertNotIn(var, self.model.variables)
        self.assertEqual(var.problem, None)

    def test_add_constraints(self):
        x = self.interface.Variable('x', type='binary')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=0., ub=3, type='integer')
        constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        constr2 = self.interface.Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
        constr3 = self.interface.Constraint(2.333 * x + y + z, ub=100.33, lb=-300)
        constr4 = self.interface.Constraint(77 * x, lb=10, name='Mul_constraint')
        constr5 = self.interface.Constraint(x, ub=-10, name='Only_var_constraint')
        constr6 = self.interface.Constraint(3, ub=88., name='Number_constraint')
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

    def test_remove_constraints(self):
        x = self.interface.Variable('x', type='binary')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=4, ub=4, type='integer')
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
        x = self.interface.Variable('x', type='binary')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=3, ub=3, type='integer')
        with self.assertRaises(ValueError):
            constraint = self.interface.Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
            self.model.add(constraint)
            self.model.update()

    @abc.abstractmethod
    def test_change_of_constraint_is_reflected_in_low_level_solver(self):
        pass

    @abc.abstractmethod
    def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
        pass

    @abc.abstractmethod
    def test_change_of_objective_is_reflected_in_low_level_solver(self):
        pass

    @abc.abstractmethod
    def test_change_variable_bounds(self):
        pass

    def test_change_variable_type(self):
        self.model.variables[-1].type = "integer"
        self.assertEqual(self.model.variables[-1].type, "integer")

    @abc.abstractmethod
    def test_change_constraint_bounds(self):
        pass

    @abc.abstractmethod
    def test_initial_objective(self):
        pass

    def test_optimize(self):
        self.model.optimize()
        self.assertEqual(self.model.status, 'optimal')
        self.assertAlmostEqual(self.model.objective.value, 0.8739215069684303)

    def test_optimize_milp(self):
        with open(TESTMILPMODELPATH) as infile:
            milp_model = self.interface.Model.from_json(json.load(infile))
        milp_model.optimize()
        self.assertEqual(milp_model.status, 'optimal')
        self.assertAlmostEqual(milp_model.objective.value, 122.5)
        for variable in milp_model.variables:
            if variable.type == 'integer':
                self.assertEqual(variable.primal % 1, 0)

    def test_change_objective(self):
        v1, v2 = self.model.variables.values()[0:2]
        self.model.objective = self.interface.Objective(1. * v1 + 1. * v2)
        self.assertEqual(str(self.model.objective), 'Maximize\n1.0*R_PGK + 1.0*R_Biomass_Ecoli_core_w_GAM')
        self.model.objective = self.interface.Objective(v1 + v2)
        self.assertEqual(str(self.model.objective), 'Maximize\n1.0*R_PGK + 1.0*R_Biomass_Ecoli_core_w_GAM')

    def test_number_objective(self):
        self.model.objective = self.interface.Objective(0.)
        self.assertEqual(self.model.objective.expression, 0)
        self.assertEqual(self.model.objective.direction, "max")
        self.assertEqual(self.model.optimize(), "optimal")

    def test_raise_on_non_linear_objective(self):
        """Test that an exception is raised when a non-linear objective is added to the model."""
        v1, v2 = self.model.variables.values()[0:2]
        self.assertRaises(ValueError, self.interface.Objective, v1 * v2 ** 3)

    @abc.abstractmethod
    def test_iadd_objective(self):
        pass

    @abc.abstractmethod
    def test_imul_objective(self):
        pass

    @abc.abstractmethod
    def test_set_copied_objective(self):
        pass

    @abc.abstractmethod
    def test_timeout(self):
        pass

    @abc.abstractmethod
    def test_set_linear_coefficients_objective(self):
        pass

    def test_instantiating_model_with_different_solver_problem_raises(self):
        self.assertRaises(TypeError, self.interface.Model, problem='Chicken soup')

    @abc.abstractmethod
    def test_set_linear_coefficients_constraint(self):
        pass

    def test_primal_values(self):
        self.model.optimize()
        primals = self.model.primal_values
        for var in self.model.variables:
            self.assertEqual(var.primal, primals[var.name])
        self.assertEqual(set(var.name for var in self.model.variables), set(primals))

    def test_reduced_costs(self):
        self.model.optimize()
        reduced_costs = self.model.reduced_costs
        for var in self.model.variables:
            self.assertEqual(var.dual, reduced_costs[var.name])
        self.assertEqual(set(var.name for var in self.model.variables), set(reduced_costs))

    def test_dual_values(self):
        self.model.optimize()
        constraint_primals = self.model.constraint_values  # TODO Fix this method name
        for constraint in self.model.constraints:
            self.assertEqual(constraint.primal, constraint_primals[constraint.name])
        self.assertEqual(set(const.name for const in self.model.constraints), set(constraint_primals))

    def test_shadow_prices(self):
        self.model.optimize()
        shadow_prices = self.model.shadow_prices
        for constraint in self.model.constraints:
            self.assertEqual(constraint.dual, shadow_prices[constraint.name])
        self.assertEqual(set(const.name for const in self.model.constraints), set(shadow_prices))

    def test_change_objective_can_handle_removed_vars(self):
        self.model.objective = self.interface.Objective(self.model.variables[0])
        self.model.remove(self.model.variables[0])
        self.model.update()
        self.model.objective = self.interface.Objective(self.model.variables[1] * 2)

    def test_clone_model_with_json(self):
        self.assertEquals(self.model.configuration.verbosity, 0)
        self.model.configuration.verbosity = 3
        self.model.optimize()
        opt = self.model.objective.value
        cloned_model = self.interface.Model.clone(self.model)
        self.assertEquals(cloned_model.configuration.verbosity, 3)
        self.assertEquals(len(cloned_model.variables), len(self.model.variables))
        self.assertEquals(len(cloned_model.constraints), len(self.model.constraints))
        cloned_model.optimize()
        self.assertAlmostEqual(cloned_model.objective.value, opt)

    def test_clone_model_with_lp(self):
        self.assertEquals(self.model.configuration.verbosity, 0)
        self.model.configuration.verbosity = 3
        self.model.optimize()
        opt = self.model.objective.value
        cloned_model = self.interface.Model.clone(self.model, use_lp=True)
        self.assertEquals(cloned_model.configuration.verbosity, 3)
        self.assertEquals(len(cloned_model.variables), len(self.model.variables))
        self.assertEquals(len(cloned_model.constraints), len(self.model.constraints))
        cloned_model.optimize()
        self.assertAlmostEqual(cloned_model.objective.value, opt)

    def test_clone_model_without_json(self):
        self.assertEquals(self.model.configuration.verbosity, 0)
        self.model.configuration.verbosity = 3
        self.model.optimize()
        opt = self.model.objective.value
        cloned_model = self.interface.Model.clone(self.model, use_json=False)
        self.assertEquals(cloned_model.configuration.verbosity, 3)
        self.assertEquals(len(cloned_model.variables), len(self.model.variables))
        self.assertEquals(len(cloned_model.constraints), len(self.model.constraints))
        cloned_model.optimize()
        self.assertAlmostEqual(cloned_model.objective.value, opt)

    def test_remove_variable_not_in_model_raises(self):
        var = self.interface.Variable("test")
        self.assertRaises(Exception, self.model._remove_variables, [var])

    def test_objective_set_linear_coefficients(self):
        x = self.interface.Variable("x", lb=0)
        y = self.interface.Variable("y", lb=0)
        c1 = self.interface.Constraint((y + 2 * (x - 3)).expand(), ub=0)
        c2 = self.interface.Constraint(y + (1 / 2) * x - 3, ub=0)
        obj = self.interface.Objective(x)
        model = self.interface.Model()
        model.add([c1, c2])
        model.objective = obj

        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 3)
        self.assertAlmostEqual(y.primal, 0)

        obj.set_linear_coefficients({y: 1})
        self.assertEqual(obj.expression - (x + y), 0)
        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 2)
        self.assertAlmostEqual(y.primal, 2)

        obj.set_linear_coefficients({x: 0})
        self.assertEqual(obj.expression - y, 0)
        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 0)
        self.assertAlmostEqual(y.primal, 3)

    def test_constraint_set_linear_coefficients(self):
        x = self.interface.Variable("x", lb=0, ub=1000)
        y = self.interface.Variable("y", lb=0)
        c1 = self.interface.Constraint(y, ub=1)
        obj = self.interface.Objective(x)
        model = self.interface.Model()
        model.add([c1])
        model.objective = obj

        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, x.ub)

        c1.set_linear_coefficients({x: 1})
        self.assertEqual(c1.expression - (x + y), 0)
        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 1)

        c1.set_linear_coefficients({x: 2})
        self.assertEqual(c1.expression - (2 * x + y), 0)
        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 0.5)


@six.add_metaclass(abc.ABCMeta)
class AbstractConfigurationTestCase(unittest.TestCase):
    @abc.abstractmethod
    def setUp(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class AbstractQuadraticProgrammingTestCase(unittest.TestCase):
    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def test_convex_obj(self):
        pass

    @abc.abstractmethod
    def test_non_convex_obj(self):
        pass

    @abc.abstractmethod
    def test_qp_convex(self):
        pass

    @abc.abstractmethod
    def test_qp_non_convex(self):
        pass
