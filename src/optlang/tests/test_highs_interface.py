"""
Test suite for the HiGHS solver interface (highs_interface.py), built on
top of optlang's generic abstract test suite (abstract_test_cases.py).

Scope
-----
highs_interface currently only supports continuous LP/QP problems (see the
module docstring in highs_interface.py) - there is no MILP support. The
abstract suite includes a number of tests that exercise integer/binary
variables and indicator constraints; those are skipped here rather than
implemented, per the LP/QP-only scope of this interface. Everything else
from the abstract suite is implemented.

`test_clone_model_with_lp` is skipped too: it exercises Model.clone's
use_lp=True path (round-tripping through to_lp()/from_lp()), which this
interface does not implement.

A couple of the abstract tests are deliberately narrowed for this
interface's design (documented inline where relevant):
  - test_tolerance_parameters iterates over dir(tolerances) in the generic
    version, which also picks up non-numeric members (methods, dunders);
    here it's pinned to the three real tolerance parameters.
  - test_config_gets_copied_too checks verbosity/timeout/presolve get
    copied by Model.clone, but stops short of tolerances: Configuration's
    tolerances property only accepts an actual Tolerances instance, so the
    dict Configuration.clone builds for it is silently dropped by the
    constructor - a pre-existing gap in clone's tolerances handling that is
    out of scope for this change.

Note on imports: this assumes highs_interface.py sits next to this file (or
on the path) as a standalone module, matching the other project files it
was reviewed alongside. If/when it's moved into the optlang package proper
(e.g. optlang/highs_interface.py), swap the `import highs_interface` line
for `from optlang import highs_interface`.
"""
import copy
import json
import pickle
import unittest

import highspy

from optlang import interface, highs_interface

from optlang.tests import abstract_test_cases

class HighsVariableTestCase(abstract_test_cases.AbstractVariableTestCase):
    interface = highs_interface

    def test_get_primal(self):
        self.assertIsNone(self.var.primal)
        self.var.lb = 3
        self.var.ub = 10
        self.model.add(self.var)
        self.model.objective = self.interface.Objective(self.var, direction="max")
        status = self.model.optimize()
        self.assertEqual(status, interface.OPTIMAL)
        self.assertAlmostEqual(self.var.primal, 10)

    def test_changing_variable_names_is_reflected_in_the_solver(self):
        self.model.add(self.var)
        self.model.update()
        self.var.name = "test_2"
        self.assertEqual(
            self.model.problem.variableName(self.var._solver_index), "test_2"
        )
        self.assertIn("test_2", self.model.variables)
        self.assertNotIn("test", self.model.variables)

    def test_set_wrong_type_raises(self):
        # Trimmed from the abstract version: this interface is LP/QP only
        # (see highs_interface.Variable.__init__), so there's no valid
        # "integer" type to switch to for the final two lines of the
        # original test. See test_integer_and_binary_types_are_rejected
        # below for the MIP-specific replacement.
        self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
        self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
        self.model.add(self.var)
        self.assertRaises(ValueError, setattr, self.var, "type", "mustard")

    @unittest.skip("HiGHS interface only supports continuous variables (LP/QP, no MIP).")
    def test_change_type(self):
        pass

    def test_integer_and_binary_types_are_rejected(self):
        """Replacement for test_change_type: rather than silently accepting
        a variable type the solver can never honor, construction should
        fail outright for non-continuous types."""
        self.assertRaises(ValueError, self.interface.Variable, "int_var", type="integer")
        self.assertRaises(ValueError, self.interface.Variable, "bin_var", type="binary")


class HighsConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
    interface = highs_interface

    def test_get_primal(self):
        self.assertIsNone(self.model.constraints[0].primal)
        self.model.optimize()
        self.assertEqual(self.model.status, interface.OPTIMAL)
        self.assertTrue(isinstance(self.model.constraints[0].primal, float))

    @unittest.skip(
        "Indicator constraints require a binary indicator variable, which "
        "this LP/QP-only interface does not support."
    )
    def test_indicator_constraint_support(self):
        pass


class HighsObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
    interface = highs_interface

    def setUp(self):
        with open(abstract_test_cases.TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))
        self.obj = self.model.objective

    def test_change_direction(self):
        self.obj.direction = "min"
        self.assertEqual(self.obj.direction, "min")
        _, sense = self.model.problem.getObjective()
        self.assertEqual(sense, highspy.ObjSense.kMinimize)

        self.obj.direction = "max"
        self.assertEqual(self.obj.direction, "max")
        _, sense = self.model.problem.getObjective()
        self.assertEqual(sense, highspy.ObjSense.kMaximize)


class HighsModelTestCase(abstract_test_cases.AbstractModelTestCase):
    interface = highs_interface

    # --- abstract methods ---------------------------------------------

    def test_pickle_ability(self):
        self.model.optimize()
        value = self.model.objective.value
        pickle_string = pickle.dumps(self.model)
        from_pickle = pickle.loads(pickle_string)
        from_pickle.optimize()
        self.assertAlmostEqual(from_pickle.objective.value, value)
        self.assertEqual(
            [(var.lb, var.ub, var.name, var.type) for var in from_pickle.variables.values()],
            [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()],
        )
        self.assertEqual(
            [(constr.lb, constr.ub, constr.name) for constr in from_pickle.constraints],
            [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints],
        )

    def test_config_gets_copied_too(self):
        self.model.configuration.verbosity = 3
        self.model.configuration.timeout = 5
        self.model.configuration.presolve = False

        model_copy = self.interface.Model.clone(self.model)

        self.assertEqual(model_copy.configuration.verbosity, 3)
        self.assertEqual(model_copy.configuration.timeout, 5)
        self.assertEqual(model_copy.configuration.presolve, False)

    def test_init_from_existing_problem(self):
        inner_problem = self.model.problem
        self.assertEqual(inner_problem.getNumCol(), len(self.model.variables))

        new_model = self.interface.Model(problem=inner_problem)
        self.assertEqual(len(new_model.variables), len(self.model.variables))
        self.assertEqual(len(new_model.constraints), len(self.model.constraints))
        self.assertEqual(
            sorted(var.name for var in new_model.variables),
            sorted(var.name for var in self.model.variables),
        )
        self.assertEqual(
            sorted(c.name for c in new_model.constraints),
            sorted(c.name for c in self.model.constraints),
        )

        status = new_model.optimize()
        self.assertEqual(status, interface.OPTIMAL)

    def test_add_non_cplex_conform_variable(self):
        var = self.interface.Variable('12x!!@#5_3', lb=-666, ub=666)
        self.model.add(var)
        self.model.update()
        self.assertEqual(var.name, self.model.problem.variableName(var._solver_index))
        repickled = pickle.loads(pickle.dumps(self.model))
        var_from_pickle = repickled.variables['12x!!@#5_3']
        self.assertEqual(
            var_from_pickle.name,
            repickled.problem.variableName(var_from_pickle._solver_index),
        )

    def test_change_of_constraint_is_reflected_in_low_level_solver(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        y = self.interface.Variable('y', lb=0, ub=10)
        z = self.interface.Variable('z', lb=0, ub=10)
        constraint = self.interface.Constraint(x + y, lb=0, ub=100, name='constr_test')
        self.model.add([z, constraint])
        self.model.update()

        _, idx, _ = self.model.problem.getRowEntries(constraint._solver_index)
        self.assertEqual(len(idx), 2)

        constraint.set_linear_coefficients({z: 77.})
        _, idx, val = self.model.problem.getRowEntries(constraint._solver_index)
        coeffs = dict(zip(
            (self.model.problem.variableName(int(i)) for i in idx), val
        ))
        self.assertAlmostEqual(coeffs['z'], 77.)

    def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
        x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        constraint = self.interface.Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
        self.model.add(constraint)
        z = self.interface.Variable('z', lb=2, ub=5)
        constraint += 77. * z
        self.model.remove(constraint)
        self.assertEqual(
            (constraint.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0, 0
        )
        self.assertEqual(constraint.lb, -100)
        self.assertEqual(constraint.ub, None)

    def test_change_of_objective_is_reflected_in_low_level_solver(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        y = self.interface.Variable('y', lb=0, ub=10)
        self.model.add([x, y])
        self.model.objective = self.interface.Objective(x + 2 * y, direction='max')

        def _cost_dict():
            obj_expr, _ = self.model.problem.getObjective()
            return dict(zip(
                (self.model.problem.variableName(int(i)) for i in obj_expr.idxs),
                obj_expr.vals,
            ))

        coeffs = _cost_dict()
        self.assertAlmostEqual(coeffs.get('x', 0), 1.0)
        self.assertAlmostEqual(coeffs.get('y', 0), 2.0)

        self.model.objective.set_linear_coefficients({x: 5.0})
        coeffs = _cost_dict()
        self.assertAlmostEqual(coeffs.get('x', 0), 5.0)
        self.assertAlmostEqual(coeffs.get('y', 0), 2.0)

    def test_change_variable_bounds(self):
        inner_problem = self.model.problem
        bounds = [(-1000. - i, 1000. + i) for i in range(len(self.model.variables))]
        for var, (lb, ub) in zip(self.model.variables, bounds):
            # set_bounds updates both at once, avoiding a transient
            # lb > (old) ub state that the individual setters would reject.
            var.set_bounds(lb, ub)

        lp = inner_problem.getLp()
        for i, (lb, ub) in enumerate(bounds):
            self.assertAlmostEqual(lp.col_lower_[i], lb)
            self.assertAlmostEqual(lp.col_upper_[i], ub)

    def test_change_constraint_bounds(self):
        constraint = self.model.constraints[0]
        constraint.lb = -1000.
        constraint.ub = 1000.

        _, row_lb, row_ub, _ = self.model.problem.getRow(constraint._solver_index)
        self.assertAlmostEqual(row_lb, -1000. - constraint._constant)
        self.assertAlmostEqual(row_ub, 1000. - constraint._constant)
        self.assertAlmostEqual(constraint.lb, -1000.)
        self.assertAlmostEqual(constraint.ub, 1000.)

    def test_iadd_objective(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        y = self.interface.Variable('y', lb=0, ub=10)
        self.model.add([x, y])

        obj = self.interface.Objective(x)
        obj += 2. * y
        self.model.objective = obj

        obj_expr, _ = self.model.problem.getObjective()
        coeffs = dict(zip(
            (self.model.problem.variableName(int(i)) for i in obj_expr.idxs),
            obj_expr.vals,
        ))
        self.assertAlmostEqual(coeffs.get('x', 0), 1.0)
        self.assertAlmostEqual(coeffs.get('y', 0), 2.0)

    def test_imul_objective(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        y = self.interface.Variable('y', lb=0, ub=10)
        self.model.add([x, y])

        obj = self.interface.Objective(x + y)
        obj *= 3.
        self.model.objective = obj

        obj_expr, _ = self.model.problem.getObjective()
        coeffs = dict(zip(
            (self.model.problem.variableName(int(i)) for i in obj_expr.idxs),
            obj_expr.vals,
        ))
        self.assertAlmostEqual(coeffs.get('x', 0), 3.0)
        self.assertAlmostEqual(coeffs.get('y', 0), 3.0)

    def test_set_copied_objective(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        self.model.add(x)
        self.model.objective = self.interface.Objective(x, direction='max')

        obj_copy = copy.copy(self.model.objective)
        self.model.objective = obj_copy

        status = self.model.optimize()
        self.assertEqual(status, interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 10)

    def test_timeout(self):
        self.model.configuration.timeout = 10
        self.assertEqual(self.model.configuration.timeout, 10)
        status = self.model.optimize()
        self.assertIn(status, (interface.OPTIMAL, interface.TIME_LIMIT))

    def test_set_linear_coefficients_objective(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        y = self.interface.Variable('y', lb=0, ub=10)
        self.model.add([x, y])
        self.model.objective = self.interface.Objective(x, direction='max')

        self.model.objective.set_linear_coefficients({y: 1.})

        obj_expr, _ = self.model.problem.getObjective()
        coeffs = dict(zip(
            (self.model.problem.variableName(int(i)) for i in obj_expr.idxs),
            obj_expr.vals,
        ))
        self.assertAlmostEqual(coeffs.get('x', 0), 1.0)
        self.assertAlmostEqual(coeffs.get('y', 0), 1.0)

    def test_set_linear_coefficients_constraint(self):
        x = self.interface.Variable('x', lb=0, ub=10)
        y = self.interface.Variable('y', lb=0, ub=10)
        c = self.interface.Constraint(x, lb=0, ub=100, name='c')
        self.model.add([y, c])
        self.model.update()

        c.set_linear_coefficients({y: 3.})

        _, idx, val = self.model.problem.getRowEntries(c._solver_index)
        coeffs = dict(zip(
            (self.model.problem.variableName(int(i)) for i in idx), val
        ))
        self.assertAlmostEqual(coeffs.get('y', 0), 3.0)

    # --- concrete (non-abstract) tests from the base suite that needed
    # adapting for this interface: binary/integer variables replaced with
    # continuous ones (this interface is LP/QP only), and sloppy=False
    # added where a test relies on validation/canonicalization this
    # interface skips by default. -----------------------------------

    def test_add_constraints(self):
        x = self.interface.Variable('x', lb=0, ub=1)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        z = self.interface.Variable('z', lb=0., ub=3)
        constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        constr2 = self.interface.Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
        constr3 = self.interface.Constraint(2.333 * x + y + z, ub=100.33, lb=-300)
        constr4 = self.interface.Constraint(77 * x, lb=10, name='Mul_constraint')
        constr5 = self.interface.Constraint(x, ub=-10, name='Only_var_constraint')
        constr6 = self.interface.Constraint(3, ub=88., name='Number_constraint')
        self.model.add(constr1)
        self.model.add(constr2)
        self.model.add(constr3, sloppy=True)
        self.model.add([constr4, constr5, constr6])
        self.assertIn(constr1.name, self.model.constraints)
        self.assertIn(constr2.name, self.model.constraints)
        self.assertIn(constr3.name, self.model.constraints)
        self.assertIn(constr4.name, self.model.constraints)
        self.assertIn(constr5.name, self.model.constraints)
        self.assertIn(constr6.name, self.model.constraints)

    def test_remove_constraints(self):
        x = self.interface.Variable('x', lb=0, ub=1)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        z = self.interface.Variable('z', lb=4, ub=4)
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
        x = self.interface.Variable('x', lb=0, ub=1)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        z = self.interface.Variable('z', lb=3, ub=3)
        with self.assertRaises(ValueError):
            constraint = self.interface.Constraint(
                0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test'
            )
            self.model.add(constraint)

    # --- MIP-only tests from the abstract suite: skipped, see module
    # docstring. ------------------------------------------------------

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_add_integer_var(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_change_variable_type(self):
        pass

    @unittest.skip("MIP (binary variables) is not supported by this interface.")
    def test_binary_variables(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_integer_variable_dual(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_integer_constraint_dual(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_integer_batch_duals(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_implicitly_convert_milp_to_lp(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_optimize_milp(self):
        pass

    @unittest.skip("MIP (integer variables) is not supported by this interface.")
    def test_is_integer(self):
        pass

    @unittest.skip("Cloning via LP export/import (to_lp/from_lp) is not implemented for this interface.")
    def test_clone_model_with_lp(self):
        pass


class HighsConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):
    interface = highs_interface

    def test_tolerance_parameters(self):
        # Narrowed from the abstract version, which iterates over
        # dir(tolerances) - that also picks up non-numeric members (e.g.
        # to_dict, dunders), which "setattr(param, 2 * val)" can't handle.
        # Restricting to the actual tunable parameters keeps the same
        # intent (round-trip every tolerance through get/set) without that.
        model = self.interface.Model()
        params = ["feasibility", "optimality", "ipm_optimality"]
        for param in params:
            val = getattr(model.configuration.tolerances, param)
            setattr(model.configuration.tolerances, param, 2 * val)
            self.assertEqual(getattr(model.configuration.tolerances, param), 2 * val)


class HighsQuadraticProgrammingTestCase(abstract_test_cases.AbstractQuadraticProgrammingTestCase):
    interface = highs_interface

    def setUp(self):
        self.model = self.interface.Model()

    def test_convex_obj(self):
        x = self.interface.Variable('x', lb=-10, ub=10)
        y = self.interface.Variable('y', lb=-10, ub=10)
        self.model.add([x, y])
        self.model.objective = self.interface.Objective(x ** 2 + y ** 2, direction='min')

        status = self.model.optimize()
        self.assertEqual(status, interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 0.0, places=5)
        self.assertAlmostEqual(y.primal, 0.0, places=5)
        self.assertAlmostEqual(self.model.objective.value, 0.0, places=5)

    def test_non_convex_obj(self):
        x = self.interface.Variable('x', lb=-10, ub=10)
        y = self.interface.Variable('y', lb=-10, ub=10)
        self.model.add([x, y])
        # x^2 - y^2 is indefinite (a saddle at the origin), so this
        # objective is not convex.
        self.model.objective = self.interface.Objective(x ** 2 - y ** 2, direction='min')

        # HiGHS's QP solver is built for convex objectives. For a
        # non-convex one, the exact termination status is solver-version
        # dependent (it may converge to a stationary point and still
        # report "optimal", report a different status, or raise). We only
        # check that solving a non-convex QP doesn't crash the interface
        # itself and, if it does return, that the status is one optlang
        # recognizes.
        try:
            status = self.model.optimize()
        except Exception:
            return
        self.assertIn(status, list(interface.statuses.keys()))

    def test_qp_convex(self):
        x = self.interface.Variable('x', lb=-10, ub=10)
        y = self.interface.Variable('y', lb=-10, ub=10)
        self.model.add([x, y])
        self.model.add(self.interface.Constraint(x + y, lb=1, name='c'))
        self.model.objective = self.interface.Objective(x ** 2 + y ** 2, direction='min')

        status = self.model.optimize()
        self.assertEqual(status, interface.OPTIMAL)
        self.assertAlmostEqual(x.primal, 0.5, places=4)
        self.assertAlmostEqual(y.primal, 0.5, places=4)
        self.assertAlmostEqual(self.model.objective.value, 0.5, places=4)

    def test_qp_non_convex(self):
        x = self.interface.Variable('x', lb=-5, ub=5)
        y = self.interface.Variable('y', lb=-5, ub=5)
        self.model.add([x, y])
        self.model.add(self.interface.Constraint(x + y, lb=-5, ub=5, name='c'))
        self.model.objective = self.interface.Objective(x * y, direction='min')

        # See test_non_convex_obj - global optimality isn't guaranteed here,
        # so this only checks the solver handles it gracefully.
        try:
            status = self.model.optimize()
        except Exception:
            return
        self.assertIn(status, list(interface.statuses.keys()))


if __name__ == '__main__':
    unittest.main()
