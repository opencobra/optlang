# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import copy
import os
import pickle
import random
import unittest


try:  # noqa: C901
    import osqp
except ImportError as e:

    if str(e).find('osqp') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise
else:

    import json

    import numpy as np
    import six
    from numpy.testing import assert_allclose

    from optlang import interface, osqp_interface
    from optlang.exceptions import SolverError
    from optlang.interface import INFEASIBLE, OPTIMAL, SPECIAL
    from optlang.osqp_interface import Constraint, Model, Objective, Variable
    from optlang.tests import abstract_test_cases

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
    TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')
    LARGE_QPS = [
        os.path.join(os.path.dirname(__file__), p)
        for p in ('data/QPLIB_8785.json', 'data/QPLIB_8938.json')]
    # taken from http://qplib.zib.de/instances.html
    LARGE_QPS = dict(zip(
        LARGE_QPS,
        ((10399, 11362, 7867.4911490000004051),
         (4001, 11999, -35.7794529499999996))))


    class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
        __test__ = True

        interface = osqp_interface

        def setUp(self):
            self.var = self.interface.Variable('test')
            dummy = self.interface.Variable('dummy')
            self.model = self.interface.Model()
            self.model.add([dummy])

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            with open(TESTMODELPATH) as tp:
                model = Model.from_lp(tp.read())
            print(model.problem.settings)
            model.optimize()
            self.assertEqual(model.status, 'optimal')
            assert_allclose(model.objective.value, 0.8739215069684305, 1e-4, 1e-4)
            ref_sol = [0.8739215069684306, -16.023526143167608, 16.023526143167604]
            primals = [var.primal for var in model.variables[0:3]]
            assert_allclose(primals, ref_sol, 1e-4, 1e-4)

        def test_get_dual(self):
            with open(TESTMODELPATH) as tp:
                model = Model.from_lp(tp.read())
            model.optimize()
            self.assertEqual(model.status, 'optimal')
            assert_allclose(model.objective.value, 0.8739215069684305, 1e-4, 1e-4)
            self.assertTrue(isinstance(model.variables[0].dual, float))

        def test_changing_variable_names_is_reflected_in_the_solver(self):
            with open(TESTMODELPATH) as tp:
                model = Model.from_lp(tp.read())
            for i, variable in enumerate(model.variables):
                old_name = variable.name
                variable.name = "var" + str(i)
                self.assertEqual(variable.name, "var" + str(i))
                self.assertIn("var" + str(i), model.problem.variables)
                self.assertIn("var" + str(i), model._variables_to_constraints_mapping)
                self.assertNotIn(old_name, model.problem.variables)
                self.assertNotIn(old_name, model.problem.variable_lbs)
                self.assertNotIn(old_name, model.problem.variable_ubs)
                self.assertNotIn(old_name, model._variables_to_constraints_mapping)

        def test_osqp_setting_bounds(self):
            with open(TESTMODELPATH) as tp:
                model = Model.from_lp(tp.read())
            var = model.variables[0]
            var.lb = 1
            self.assertEqual(var.lb, 1)
            model.update()
            self.assertEqual(model.problem.variable_lbs[var.name], 1)
            var.ub = 2
            self.assertEqual(var.ub, 2)
            model.update()
            self.assertEqual(model.problem.variable_ubs[var.name], 2)

        def test_set_bounds_to_none(self):
            model = self.model
            var = self.var

            model.objective = self.interface.Objective(1.0 * var)
            model.update()
            print(model.problem.variables)
            self.assertEqual(model.optimize(), interface.INFEASIBLE)
            var.ub = 10
            self.assertEqual(model.optimize(), interface.OPTIMAL)
            var.ub = None
            self.assertEqual(model.optimize(), interface.INFEASIBLE)
            self.model.objective.direction = "min"
            var.lb = -10
            self.assertEqual(model.optimize(), interface.OPTIMAL)
            var.lb = None
            self.assertEqual(model.optimize(), interface.INFEASIBLE)

        def test_set_bounds_method(self):
            var = self.interface.Variable("test", lb=-10)
            c = self.interface.Constraint(var, lb=-100)
            # OSQP needs at least two variables
            model = self.interface.Model()
            obj = self.interface.Objective(1.0 * var)
            model.add([c])
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
            self.assertEqual(model.optimize(), interface.INFEASIBLE)

            self.assertRaises(ValueError, var.set_bounds, 2, 1)

        def test_set_wrong_type_raises(self):
            self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
            self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
            self.model.add(self.var)
            self.model.update()
            self.assertRaises(ValueError, setattr, self.var, "type", "mustard")
            self.var.type = "continuous"
            self.assertEqual(self.var.type, "continuous")

        def test_change_variable_type(self):
            # not supported by OSQP
            pass



    class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
        interface = osqp_interface

        def test_set_linear_coefficients(self):
            self.model.add(self.constraint)
            self.constraint.set_linear_coefficients({Variable('chip'): 33., self.model.variables.R_PGK: -33})
            coefs = {
                v.name: self.model.problem.constraint_coefs[(self.constraint.name, v.name)]
                for v in self.model.variables
                if (self.constraint.name, v.name) in self.model.problem.constraint_coefs
            }
            self.assertEqual(coefs,
                             dict([('R_PGK', -33.0), ('chap', 1.0), ('chip', 33.0)]))

        def test_get_primal(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            assert_allclose(self.model.objective.value, 0.8739215069684305, 1e-3, 1e-3)
            primals = [constraint.primal for constraint in self.model.constraints]
            assert_allclose(primals, np.zeros(len(primals)), 1e-3, 1e-3)  # only equality constraints

        def test_get_dual(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            assert_allclose(self.model.objective.value, 0.8739215069684305, 1e-3, 1e-3)
            duals = [constraint.dual for constraint in self.model.constraints]
            self.assertTrue(all(isinstance(d, float) for d in duals))

        def test_set_constraint_bounds_to_none(self):
            model = self.interface.Model()
            var = self.interface.Variable("test")
            const = self.interface.Constraint(var, lb=-10, ub=10)
            obj = self.interface.Objective(var)
            model.add(const)
            model.objective = obj
            self.assertEqual(model.optimize(), interface.OPTIMAL)
            const.ub = None
            self.assertEqual(model.optimize(), interface.INFEASIBLE)
            const.ub = 10
            const.lb = None
            obj.direction = "min"
            self.assertEqual(model.optimize(), interface.INFEASIBLE)


    class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
        interface = osqp_interface

        def setUp(self):
            with open(TESTMODELPATH) as tp:
                model = Model.from_lp(tp.read())
            self.obj = model.objective
            self.model = model

        def test_change_direction(self):
            self.obj.direction = "min"
            self.assertEqual(self.obj.direction, "min")
            self.assertEqual(self.model.problem.direction, 1)

            self.obj.direction = "max"
            self.assertEqual(self.obj.direction, "max")
            self.assertEqual(self.model.problem.direction, -1)


    class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
        interface = osqp_interface

        def test_change_variable_type(self):
            pass

        def test_clone_model_with_lp(self):
            pass

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
            self.assertEqual(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 1
            model_copy = copy.copy(self.model)
            self.assertEqual(model_copy.configuration.verbosity, 1)

        def test_init_from_existing_problem(self):
            inner_prob = self.model.problem
            self.assertEqual(len(self.model.variables), len(inner_prob.variables))
            self.assertEqual(len(self.model.constraints), len(inner_prob.constraints))
            self.assertEqual(set(self.model.variables.keys()), inner_prob.variables)
            self.assertEqual(set(self.model.constraints.keys()), inner_prob.constraints)

        def test_add_non_cplex_conform_variable(self):
            var = Variable('12x!!@#5_3', lb=-666, ub=666)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertIn(var.name, self.model.problem.variables)
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            print(repickled.variables)
            var_from_pickle = repickled.variables['12x!!@#5_3']
            self.assertIn(var_from_pickle.name, self.model.problem.variables)

        def test_osqp_remove_variable(self):
            var = self.model.variables[0]
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.model.update()
            self.assertNotIn(var.name, self.model.problem.variables)
            self.assertNotIn(var.name, self.model.problem.variable_lbs)
            self.assertNotIn(var.name, self.model.problem.variable_ubs)
            self.assertEqual(var.problem, None)

        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            self.assertEqual(
                (self.model.constraints['test'].expression - (0.4 * y + 0.3 * x)).expand() - 0,
                0
            )
            self.assertEqual(
                [self.model.problem.constraint_coefs[k]
                    for k in [('test', 'x'), ('test', 'y')]],
                [0.3, 0.4])
            z = Variable('z', lb=3, ub=4, type='integer')
            constraint += 77. * z
            self.assertEqual(
                [self.model.problem.constraint_coefs[k]
                    for k in [('test', 'x'), ('test', 'y'), ('test', 'z')]],
                [0.3, 0.4, 77.])
            self.assertEqual(
                (self.model.constraints['test'].expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0,
                0
            )

        def test_implicitly_convert_milp_to_lp(self):
            pass

        def test_integer_batch_duals(self):
            pass

        def test_integer_constraint_dual(self):
            pass

        def test_integer_variable_dual(self):
            pass

        def test_is_integer(self):
            pass

        def test_optimize(self):
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            assert_allclose(self.model.objective.value, 0.8739215069684303, 1e-3, 1e-3)

        def test_optimize_milp(self):
            pass

        def test_non_convex_obj(self):
            pass

        def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            z = Variable('z', lb=2, ub=5, type='integer')
            constraint += 77. * z
            self.model.remove(constraint)
            self.assertEqual(constraint.name, "test")
            self.assertEqual(constraint.lb, -100)
            self.assertEqual(constraint.ub, None)
            self.assertEqual((constraint.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand(), 0.0)

        def test_change_of_objective_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            objective = Objective(0.3 * x + 0.4 * y, name='obj', direction='max')
            self.model.objective = objective
            for variable in self.model.variables:
                coeff = self.model.problem.obj_linear_coefs.get(variable.name, 0.0)
                if variable.name == 'x':
                    self.assertEqual(coeff, 0.3)
                elif variable.name == 'y':
                    self.assertEqual(coeff, 0.4)
                else:
                    self.assertEqual(coeff, 0.)
            z = Variable('z', lb=0.000003, ub=0.000003, type='continuous')
            objective += 77. * z
            for variable in self.model.variables:
                coeff = self.model.problem.obj_linear_coefs.get(variable.name, 0.0)
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
            inner_problem_bounds = [
                (inner_prob.variable_lbs[v.name],
                 inner_prob.variable_ubs[v.name])
                for v in self.model.variables
            ]
            bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertEqual(bounds, inner_problem_bounds)
            for var in self.model.variables.values():
                var.lb = random.uniform(-1000, 1000)
                var.ub = random.uniform(var.lb, 1000)
            self.model.update()
            inner_problem_bounds_new = [
                (inner_prob.variable_lbs[v.name],
                 inner_prob.variable_ubs[v.name])
                for v in self.model.variables
            ]
            bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        def test_change_constraint_bounds(self):
            constraint = self.model.constraints[0]
            value = 42
            constraint.ub = value
            self.assertEqual(constraint.ub, value)
            constraint.lb = value
            self.assertEqual(constraint.lb, value)
            self.assertEqual(self.model.problem.constraint_lbs[constraint.name], value)
            self.assertEqual(self.model.problem.constraint_ubs[constraint.name], value)

        def test_iadd_objective(self):
            v1, v2, v3 = list(self.model.variables.values())[0:3]
            self.model.objective += 2. * v2 - 3. * v3
            obj_coeff = self.model.problem.obj_linear_coefs
            self.assertEqual(obj_coeff,
                             {v1.name: 1, v2.name: 2, v3.name: -3})

        def test_imul_objective(self):
            self.model.objective *= 2.
            obj_coeff = self.model.problem.obj_linear_coefs
            self.assertEqual(obj_coeff, {self.model.variables[0].name: 2.0})

        def test_set_copied_objective(self):
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.direction, "max")
            self.assertEqual(self.model.objective.expression, 1.0 * self.model.variables["R_Biomass_Ecoli_core_w_GAM"])

        def test_timeout(self):
            self.model.configuration.timeout = 1e-6
            status = self.model.optimize()
            self.assertEqual(status, 'time_limit')

        def test_set_linear_coefficients_objective(self):
            self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
            self.assertEqual(self.model.problem.obj_linear_coefs[self.model.variables.R_TPI.name], 666.)

        def test_set_linear_coefficients_constraint(self):
            constraint = self.model.constraints.M_atp_c
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], -59.8100000000000)
            constraint.set_linear_coefficients({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], 666.)

        def test_osqp_change_objective_can_handle_removed_vars(self):
            self.model.objective = Objective(self.model.variables[0])
            self.model.remove(self.model.variables[0])
            self.model.update()
            self.model.objective = Objective(self.model.variables[1] ** 2)
            self.model.remove(self.model.variables[1])
            self.model.update()
            self.model.objective = Objective(self.model.variables[2])
            self.assertEqual(self.model.problem.obj_linear_coefs,
                {self.model.variables[2].name: 1.0})


    class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):

        interface = osqp_interface

        def setUp(self):
            self.model = Model()
            self.configuration = self.model.configuration

        def test_tolerance_parameters(self):
            model = self.interface.Model()
            params = ["optimality", "feasibility"]
            for param in params:
                val = getattr(model.configuration.tolerances, param)
                print(val)
                setattr(model.configuration.tolerances, param, 2 * val)
                self.assertEqual(
                    getattr(model.configuration.tolerances, param), 2 * val
                )

        def test_solver(self):
            for option in ("qdldl", "mkl pardiso"):
                self.configuration.linear_solver = option
                self.assertEqual(self.configuration.linear_solver, option)
                self.assertEqual(self.model.problem.settings["linsys_solver"], option)

            self.assertRaises(ValueError, setattr, self.configuration, "lp_method", "weird_stuff")

        def test_lp_method(self):
            for option in ("auto", "primal"):
                self.configuration.lp_method = option
                self.assertEqual(self.configuration.lp_method, "primal")

            self.assertRaises(ValueError, setattr, self.configuration, "lp_method", "weird_stuff")

        def test_qp_method(self):
            for option in osqp_interface._QP_METHODS:
                self.configuration.qp_method = option
                self.assertEqual(self.configuration.qp_method, "primal")

            self.assertRaises(ValueError, setattr, self.configuration, "qp_method", "weird_stuff")

        def test_verbosity(self):
            for i in range(4):
                self.model.configuration.verbosity = i
                self.assertEqual(self.model.configuration.verbosity, i)

        def test_presolve(self):
            for presolve in (True, False):
                self.configuration.presolve = presolve
                self.assertEqual(self.configuration.presolve, presolve)

            self.assertRaises(ValueError, setattr, self.configuration, "presolve", "what?")


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
            model.objective = obj
            model.optimize()
            assert_allclose(model.objective.value, 0.5, 1e-4, 1e-4)
            assert_allclose(self.x1.primal, 0.5, 1e-4, 1e-4)
            assert_allclose(self.x2.primal, 0.5, 1e-4, 1e-4)

            obj_2 = Objective(self.x1, direction="min")
            model.objective = obj_2
            model.optimize()
            assert_allclose(model.objective.value, 0.0, 1e-4, 1e-4)
            assert_allclose(self.x1.primal, 0.0, 1e-4, 1e-4)

        def test_non_convex_obj(self):
            model = self.model
            obj = Objective(self.x1 ** 2 + self.x2 ** 2, direction="max")
            model.objective = obj
            self.assertRaises(ValueError, model.optimize)

            obj_2 = Objective(self.x1, direction="min")
            model.objective = obj_2
            model.optimize()
            assert_allclose(model.objective.value, 0.0, 1e-4, 1e-4)
            assert_allclose(self.x1.primal, 0.0, 1e-4, 1e-4)

        def test_qp_convex(self):
            for qp, info in six.iteritems(LARGE_QPS):
                nv, nc, ref_sol = info
                prob = json.load(open(qp))
                model = Model.from_json(prob)
                model.configuration.tolerances.optimality = 1e-4
                model.configuration.verbosity = 1
                model.optimize()
                self.assertEqual(len(model.variables), nv)
                self.assertEqual(len(model.constraints), nc)
                self.assertEqual(model.status, OPTIMAL)
                assert_allclose(model.objective.value, ref_sol, 1e-3, 1e-3)

        def test_qp_non_convex(self):
            # unsupported by OSQP
            pass

        def test_quadratic_objective_expression(self):
            objective = Objective(self.x1 ** 2 + self.x2 ** 2, direction="min")
            self.model.objective = objective
            self.assertEqual((self.model.objective.expression - (self.x1 ** 2 + self.x2 ** 2)).simplify(), 0)
            self.assertEqual(len(self.model.problem.obj_quadratic_coefs), 2)


    class UnsolvedTestCase(unittest.TestCase):

        interface = osqp_interface

        def setUp(self):
            model = self.interface.Model()
            x = self.interface.Variable('x', lb=0, ub=10)
            constr1 = self.interface.Constraint(1. * x, lb=3, name="constr1")
            obj = self.interface.Objective(2 * x)
            model.add(x)
            model.add(constr1)
            model.objective = obj
            self.model = model
            self.continuous_var = x
            self.constraint = constr1

        def test_status(self):
            self.assertIs(self.model.status, None)

        def test_objective_value(self):
            self.assertIs(self.model.objective.value, None)

        def test_variable_primal(self):
            self.assertIs(self.continuous_var.primal, None)

        def test_variable_dual(self):
            self.assertIs(self.continuous_var.primal, None)

        def test_constraint_primal(self):
            self.assertIs(self.constraint.primal, None)

        def test_constraint_dual(self):
            self.assertIs(self.constraint.dual, None)

        def test_primal_values(self):
            with self.assertRaises(SolverError) as context:
                self.model.primal_values
            self.assertIn("not been solved", str(context.exception))

        def test_constraint_values(self):
            with self.assertRaises(SolverError) as context:
                self.model.constraint_values
            self.assertIn("not been solved", str(context.exception))

        def test_reduced_costs(self):
            with self.assertRaises(SolverError) as context:
                self.model.reduced_costs
            self.assertIn("not been solved", str(context.exception))

        def test_shadow_prices(self):
            with self.assertRaises(SolverError) as context:
                self.model.shadow_prices
            self.assertIn("not been solved", str(context.exception))
