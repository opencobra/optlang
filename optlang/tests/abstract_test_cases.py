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

import abc
import unittest

import six
from optlang import interface
import pickle
import json
import os

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

    @abc.abstractmethod
    def test_get_dual(self):
        pass

    def test_setting_lower_bound_higher_than_upper_bound_raises(self):
        self.model.add(self.var)
        self.var.ub = 0
        self.assertRaises(ValueError, setattr, self.model.variables[0], 'lb', 100.)

    def test_setting_nonnumerical_bounds_raises(self):
        self.assertRaises(TypeError, setattr, self.var, "lb", "Ministrone")
        self.model.add(self.var)
        self.assertRaises(TypeError, setattr, self.model.variables[0], 'lb', 'Chicken soup')

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
    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def test_indicator_constraint_support(self):
        pass

    @abc.abstractmethod
    def test_get_primal(self):
        pass

    @abc.abstractmethod
    def test_get_dual(self):
        pass

    @abc.abstractmethod
    def test_change_constraint_name(self):
        pass

    @abc.abstractmethod
    def test_setting_lower_bound_higher_than_upper_bound_raises(self):
        pass

    @abc.abstractmethod
    def test_setting_nonnumerical_bounds_raises(self):
        pass

    @abc.abstractmethod
    def test_set_constraint_bounds_to_none(self):
        pass


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

    @abc.abstractmethod
    def test_copy(self):
        pass

    @abc.abstractmethod
    def test_deepcopy(self):
        pass

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
        model = self.interface.Model()
        x = self.interface.Variable('x', type='binary')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=0., ub=3, type='integer')
        constr1 = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        constr2 = Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
        constr3 = Constraint(2.333 * x + y + z, ub=100.33, lb=-300)
        constr4 = Constraint(77 * x, lb=10, name='Mul_constraint')
        constr5 = Constraint(x, ub=-10, name='Only_var_constraint')
        constr6 = Constraint(3, ub=88., name='Number_constraint')
        model.add(constr1)
        model.update()
        model.add(constr2)
        model.update()
        model.add(constr3)
        model.update()
        model.add([constr4, constr5, constr6])
        model.update()
        self.assertIn(constr1.name, model.constraints)
        self.assertIn(constr2.name, model.constraints)
        self.assertIn(constr3.name, model.constraints)
        self.assertIn(constr4.name, model.constraints)
        self.assertIn(constr5.name, model.constraints)
        self.assertIn(constr6.name, model.constraints)
        self.assertEqual(
            model.problem.linear_constraints.get_coefficients((('test', 'y'), ('test', 'z'), ('test', 'x'))),
            [0.4, 66, 0.3])
        self.assertEqual(model.problem.linear_constraints.get_coefficients((('test2', 'y'), ('test2', 'x'))),
                         [1., 2.333])
        self.assertEqual(model.problem.linear_constraints.get_coefficients('Mul_constraint', 'x'), 77.)
        self.assertEqual(model.problem.linear_constraints.get_coefficients('Only_var_constraint', 'x'), 1.)

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
        constraint = self.interface.Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
        self.model.add(constraint)
        self.assertRaises(ValueError, self.model.update)

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
        self.model.objective = self.interface.Objective(self.model.variables[0])
        self.model.remove(self.model.variables[0])
        self.model.update()
        self.model.objective = self.interface.Objective(self.model.variables[1] * 2)

    def test_clone_model(self):
        self.assertEquals(self.model.configuration.verbosity, 0)
        self.model.configuration.verbosity = 3
        cloned_model = self.interface.Model.clone(self.model)
        self.assertEquals(cloned_model.configuration.verbosity, 3)
        self.assertEquals(len(cloned_model.variables), len(self.model.variables))
        self.assertEquals(len(cloned_model.constraints), len(self.model.constraints))


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
