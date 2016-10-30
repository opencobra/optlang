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

__test__ = False

@six.add_metaclass(abc.ABCMeta)
class AbstractVariableTestCase(unittest.TestCase):
    @unittest.skip('Abstract test method not implemented.')
    def test_magic(self):
        pass

    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def test_set_wrong_type_raises(self):
        pass

    @abc.abstractmethod
    def test_change_name(self):
        pass

    @abc.abstractmethod
    def test_get_primal(self):
        pass

    @abc.abstractmethod
    def test_get_dual(self):
        pass

    @abc.abstractmethod
    def test_setting_lower_bound_higher_than_upper_bound_raises(self):
        pass

    @abc.abstractmethod
    def test_setting_nonnumerical_bounds_raises(self):
        pass

    @abc.abstractmethod
    def test_changing_variable_names_is_reflected_in_the_solver(self):
        pass

    @abc.abstractmethod
    def test_setting_bounds(self):
        pass

    @abc.abstractmethod
    def test_set_bounds_to_none(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class AbstractConstraintTestCase(unittest.TestCase):
    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def test_set_linear_coefficients(self):
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
    def test_setting_bounds(self):
        pass

    @abc.abstractmethod
    def test_remove_constraint(self):
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

    @abc.abstractmethod
    def test_set_linear_objective_coefficients(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class AbstractModelTestCase(unittest.TestCase):
    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def test_create_empty_model(self):
        pass

    @abc.abstractmethod
    def test_pickle_ability(self):
        pass

    @abc.abstractmethod
    def test_pickle_empty_model(self):
        pass

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

    @abc.abstractmethod
    def test_add_variable(self):
        pass

    @abc.abstractmethod
    def test_add_integer_var(self):
        pass

    @abc.abstractmethod
    def test_add_non_cplex_conform_variable(self):
        pass

    @abc.abstractmethod
    def test_remove_variable(self):
        pass

    @abc.abstractmethod
    def test_remove_variable_str(self):
        pass

    @abc.abstractmethod
    def test_add_constraints(self):
        pass

    @abc.abstractmethod
    def test_remove_constraints(self):
        pass

    @abc.abstractmethod
    def test_add_nonlinear_constraint_raises(self):
        pass

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

    @abc.abstractmethod
    def test_change_variable_type(self):
        pass

    @abc.abstractmethod
    def test_change_constraint_bounds(self):
        pass

    @abc.abstractmethod
    def test_initial_objective(self):
        pass

    @abc.abstractmethod
    def test_optimize(self):
        pass

    @abc.abstractmethod
    def test_optimize_milp(self):
        pass

    @abc.abstractmethod
    def test_change_objective(self):
        pass

    @abc.abstractmethod
    def test_number_objective(self):
        pass

    @abc.abstractmethod
    def test_raise_on_non_linear_objective(self):
        pass

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

    @abc.abstractmethod
    def test_instantiating_model_with_different_solver_problem_raises(self):
        pass

    @abc.abstractmethod
    def test_set_linear_coefficients_constraint(self):
        pass

    @abc.abstractmethod
    def test_primal_values(self):
        pass

    @abc.abstractmethod
    def test_reduced_costs(self):
        pass

    @abc.abstractmethod
    def test_dual_values(self):
        pass

    @abc.abstractmethod
    def test_shadow_prices(self):
        pass

    @abc.abstractmethod
    def test_change_objective_can_handle_removed_vars(self):
        pass

    @abc.abstractmethod
    def test_clone_solver(self):
        pass


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