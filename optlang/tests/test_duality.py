# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
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

import unittest
from optlang.glpk_interface import Model, Variable, Constraint, Objective
from optlang.duality import convert_linear_problem_to_dual
import optlang


class DualityTestCase(unittest.TestCase):
    def setUp(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        z = Variable("z", lb=0)

        c1 = Constraint(x, ub=10)
        c2 = Constraint(y - 2 * z, lb=0, ub=0)
        c3 = Constraint(x - z, lb=3, ub=15)

        obj = Objective(x + y + z)

        model = Model()
        model.add([c1, c2, c3])
        model.objective = obj
        self.model = model
        self.c1 = c1

    def test_dual_value_equal_primal(self):
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 31)
        self.assertEqual(dual.objective.value, 31)

    def test_making_primal_unbounded(self):
        self.model.remove(self.c1)
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()
        self.assertEqual(primal_status, optlang.interface.UNBOUNDED)
        self.assertEqual(dual_status, optlang.interface.INFEASIBLE)

    def test_making_primal_infeasible(self):
        # Note: The dual of an infeasible primal will be either unbounded or infeasible. In this case it is unbounded.
        c4 = Constraint(self.model.variables["z"], lb=100)
        self.model.add(c4)
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()
        self.assertEqual(primal_status, optlang.interface.INFEASIBLE)
        self.assertEqual(dual_status, optlang.interface.UNBOUNDED)

    def test_non_standard_raises(self):
        self.model.variables[0].lb = -10
        self.assertRaises(ValueError, convert_linear_problem_to_dual, self.model)

    def test_non_linear_raises(self):
        model = optlang.interface.Model.clone(self.model)  # GLPK interface doesn't allow non-linear constraints
        c4 = optlang.interface.Constraint(2 * model.variables["x"]**2, ub=7)
        model.add(c4)
        self.assertRaises(ValueError, convert_linear_problem_to_dual, model)

    def test_non_continuous_raises(self):
        self.model.variables["z"].type = "integer"
        self.assertRaises(ValueError, convert_linear_problem_to_dual, self.model)

    def test_infinity(self):
        dual = convert_linear_problem_to_dual(self.model, infinity=1000)
        for var in dual.variables:
            self.assertTrue(var.ub <= 1000)
            self.assertTrue(var.lb >= -1000)

    def test_dual_is_standard_form(self):
        dual = convert_linear_problem_to_dual(self.model, infinity=1000)
        dual_2 = convert_linear_problem_to_dual(self.model, maintain_standard_form=False, infinity=1000)
        self.assertTrue(all(var.lb >= 0 for var in dual.variables))
        self.assertFalse(all(var.lb >= 0 for var in dual_2.variables))

    def test_variable_with_positive_lower_bound(self):
        self.model.variables["x"].lb = 1
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 31)
        self.assertEqual(dual.objective.value, 31)

    def test_variable_with_upper_bound(self):
        self.model.variables["x"].ub = 4
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 7)
        self.assertEqual(dual.objective.value, 7)

    def test_with_minimization(self):
        self.model.objective = Objective(self.model.objective.expression * -1, direction="min")
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, -31)
        self.assertEqual(dual.objective.value, -31)

    def test_equality_constraint_not_zero(self):
        c2 = self.model.constraints[1]
        c2.ub = 1
        c2.lb = 1
        dual = convert_linear_problem_to_dual(self.model, maintain_standard_form=False)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 32)
        self.assertEqual(dual.objective.value, 32)

    def test_empty_constraint(self):
        c4 = Constraint(0, lb=0, ub=13)
        self.model.add(c4)
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 31)
        self.assertEqual(dual.objective.value, 31)

    def test_free_constraint(self):
        c4 = Constraint(self.model.variables["x"], lb=None, ub=None)
        self.model.add(c4)
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 31)
        self.assertEqual(dual.objective.value, 31)

    def test_zero_bound_variable(self):
        w = Variable("w", lb=0, ub=0)
        self.model.add(w)
        dual = convert_linear_problem_to_dual(self.model)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 31)
        self.assertEqual(dual.objective.value, 31)

    def test_explicit_model(self):
        dual = Model()
        convert_linear_problem_to_dual(self.model, dual_model=dual)
        primal_status = self.model.optimize()
        dual_status = dual.optimize()

        self.assertEqual(primal_status, optlang.interface.OPTIMAL)
        self.assertEqual(dual_status, optlang.interface.OPTIMAL)

        self.assertEqual(self.model.objective.value, 31)
        self.assertEqual(dual.objective.value, 31)
