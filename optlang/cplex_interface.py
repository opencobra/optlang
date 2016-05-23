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

"""Solver interface for the IBM ILOG CPLEX Optimization Studio solver.

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""
import collections
import logging
import sys

import six
from six.moves import StringIO

log = logging.getLogger(__name__)

import tempfile
import sympy
from sympy.core.add import _unevaluated_Add
from sympy.core.mul import _unevaluated_Mul
from sympy.core.singleton import S
import cplex
from optlang import interface
from optlang.util import inheritdocstring

Zero = S.Zero
One = S.One

_CPLEX_STATUS_TO_STATUS = {
    cplex.Cplex.solution.status.MIP_abort_feasible: interface.ABORTED,
    cplex.Cplex.solution.status.MIP_abort_infeasible: interface.ABORTED,
    cplex.Cplex.solution.status.MIP_dettime_limit_feasible: interface.TIME_LIMIT,
    cplex.Cplex.solution.status.MIP_dettime_limit_infeasible: interface.TIME_LIMIT,
    cplex.Cplex.solution.status.MIP_feasible: interface.FEASIBLE,
    cplex.Cplex.solution.status.MIP_feasible_relaxed_inf: interface.SPECIAL,
    cplex.Cplex.solution.status.MIP_feasible_relaxed_quad: interface.SPECIAL,
    cplex.Cplex.solution.status.MIP_feasible_relaxed_sum: interface.SPECIAL,
    cplex.Cplex.solution.status.MIP_infeasible: interface.INFEASIBLE,
    cplex.Cplex.solution.status.MIP_infeasible_or_unbounded: interface.INFEASIBLE_OR_UNBOUNDED,
    cplex.Cplex.solution.status.MIP_optimal: interface.OPTIMAL,
    cplex.Cplex.solution.status.MIP_optimal_infeasible: interface.SPECIAL,
    cplex.Cplex.solution.status.MIP_optimal_relaxed_inf: interface.SPECIAL,
    cplex.Cplex.solution.status.MIP_optimal_relaxed_sum: interface.SPECIAL,
    cplex.Cplex.solution.status.MIP_time_limit_feasible: interface.TIME_LIMIT,
    cplex.Cplex.solution.status.MIP_time_limit_infeasible: interface.TIME_LIMIT,
    cplex.Cplex.solution.status.MIP_unbounded: interface.UNBOUNDED,
    cplex.Cplex.solution.status.abort_dettime_limit: interface.ABORTED,
    cplex.Cplex.solution.status.abort_dual_obj_limit: interface.ABORTED,
    cplex.Cplex.solution.status.abort_iteration_limit: interface.ABORTED,
    cplex.Cplex.solution.status.abort_obj_limit: interface.ABORTED,
    cplex.Cplex.solution.status.abort_primal_obj_limit: interface.ABORTED,
    cplex.Cplex.solution.status.abort_relaxed: interface.ABORTED,
    cplex.Cplex.solution.status.abort_time_limit: interface.TIME_LIMIT,
    cplex.Cplex.solution.status.abort_user: interface.ABORTED,
    cplex.Cplex.solution.status.conflict_abort_contradiction: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_dettime_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_iteration_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_memory_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_node_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_obj_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_time_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_abort_user: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_feasible: interface.SPECIAL,
    cplex.Cplex.solution.status.conflict_minimal: interface.SPECIAL,
    cplex.Cplex.solution.status.fail_feasible: interface.SPECIAL,
    cplex.Cplex.solution.status.fail_feasible_no_tree: interface.SPECIAL,
    cplex.Cplex.solution.status.fail_infeasible: interface.SPECIAL,
    cplex.Cplex.solution.status.fail_infeasible_no_tree: interface.SPECIAL,
    cplex.Cplex.solution.status.feasible: interface.FEASIBLE,
    cplex.Cplex.solution.status.feasible_relaxed_inf: interface.SPECIAL,
    cplex.Cplex.solution.status.feasible_relaxed_quad: interface.SPECIAL,
    cplex.Cplex.solution.status.feasible_relaxed_sum: interface.SPECIAL,
    cplex.Cplex.solution.status.first_order: interface.SPECIAL,
    cplex.Cplex.solution.status.infeasible: interface.INFEASIBLE,
    cplex.Cplex.solution.status.infeasible_or_unbounded: interface.INFEASIBLE_OR_UNBOUNDED,
    cplex.Cplex.solution.status.mem_limit_feasible: interface.MEMORY_LIMIT,
    cplex.Cplex.solution.status.mem_limit_infeasible: interface.MEMORY_LIMIT,
    cplex.Cplex.solution.status.node_limit_feasible: interface.NODE_LIMIT,
    cplex.Cplex.solution.status.node_limit_infeasible: interface.NODE_LIMIT,
    cplex.Cplex.solution.status.num_best: interface.NUMERIC,
    cplex.Cplex.solution.status.optimal: interface.OPTIMAL,
    cplex.Cplex.solution.status.optimal_face_unbounded: interface.SPECIAL,
    cplex.Cplex.solution.status.optimal_infeasible: interface.INFEASIBLE,
    cplex.Cplex.solution.status.optimal_populated: interface.SPECIAL,
    cplex.Cplex.solution.status.optimal_populated_tolerance: interface.SPECIAL,
    cplex.Cplex.solution.status.optimal_relaxed_inf: interface.SPECIAL,
    cplex.Cplex.solution.status.optimal_relaxed_quad: interface.SPECIAL,
    cplex.Cplex.solution.status.optimal_relaxed_sum: interface.SPECIAL,
    cplex.Cplex.solution.status.optimal_tolerance: interface.OPTIMAL,
    cplex.Cplex.solution.status.populate_solution_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.solution_limit: interface.SPECIAL,
    cplex.Cplex.solution.status.unbounded: interface.UNBOUNDED,
    cplex.Cplex.solution.status.relaxation_unbounded: interface.UNBOUNDED,
    # 102: interface.OPTIMAL # The same as cplex.Cplex.solution.status.optimal_tolerance
}

_LP_METHODS = ["auto", "primal", "dual", "network", "barrier", "sifting", "concurrent"]

_SOLUTION_TARGETS = ("auto", "convex", "local", "global")

_QP_METHODS = ("auto", "primal", "dual", "network", "barrier")

_CPLEX_VTYPE_TO_VTYPE = {'C': 'continuous', 'I': 'integer', 'B': 'binary'}

_VTYPE_TO_CPLEX_VTYPE = dict(
    [(val, key) for key, val in six.iteritems(_CPLEX_VTYPE_TO_VTYPE)]
)


def _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(lb, ub):
    """Helper function used by Constraint and Model"""
    if lb is None and ub is None:
        raise Exception("Free constraint ...")
    elif lb is None:
        sense = 'L'
        rhs = float(ub)
        range_value = 0.
    elif ub is None:
        sense = 'G'
        rhs = float(lb)
        range_value = 0.
    elif lb == ub:
        sense = 'E'
        rhs = float(lb)
        range_value = 0.
    elif lb > ub:
        raise ValueError("Lower bound is larger than upper bound.")
    else:
        sense = 'R'
        rhs = float(lb)
        range_value = float(ub - lb)
    return sense, rhs, range_value


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @interface.Variable.type.setter
    def type(self, value):
        if self.problem is not None:
            try:
                cplex_kind = _VTYPE_TO_CPLEX_VTYPE[value]
            except KeyError:
                raise Exception("CPLEX cannot handle variables of type %s. \
                            The following variable types are available:\n" +
                                " ".join(_VTYPE_TO_CPLEX_VTYPE.keys()))
            self.problem.problem.variables.set_types(self.name, cplex_kind)
        super(Variable, self).__setattr__('type', value)

    @property
    def primal(self):
        if self.problem:
            primal_from_solver = self.problem.problem.solution.get_values(self.name)
            return self._round_primal_to_bounds(primal_from_solver)
        else:
            return None

    @property
    def dual(self):
        if self.problem is not None:
            if self.problem.problem.get_problem_type() != self.problem.problem.problem_type.LP:  # cplex cannot determine reduced costs for MILP problems ...
                return None
            return self.problem.problem.solution.get_reduced_costs(self.name)
        else:
            return None

    @interface.Variable.name.setter
    def name(self, value):
        if getattr(self, "model", None) is not None:
            self.model.problem.variables.set_names(self.name, value)
        super(Variable, Variable).name.fset(self, value)


@six.add_metaclass(inheritdocstring)
class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = True

    def __init__(self, expression, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)
        if self.ub is not None and self.lb is not None and self.lb > self.ub:
            raise ValueError(
                "Lower bound %f is larger than upper bound %f in constraint %s" %
                (self.lb, self.ub, self)
            )

    def _get_expression(self):
        if self.problem is not None:
            cplex_problem = self.problem.problem
            cplex_row = cplex_problem.linear_constraints.get_rows(self.name)
            variables = self.problem._variables
            expression = sympy.Add._from_args(
                [sympy.Mul._from_args((sympy.RealNumber(cplex_row.val[i]), variables[ind])) for i, ind in
                 enumerate(cplex_row.ind)])
            self._expression = expression
        return self._expression

    def _set_coefficients_low_level(self, variables_coefficients_dict):
        self_name = self.name
        if self.is_Linear:
            cplex_format = [(self_name, variable.name, coefficient) for variable, coefficient in
                            six.iteritems(variables_coefficients_dict)]
            self.problem.problem.linear_constraints.set_coefficients(cplex_format)
        else:
            raise Exception('_set_coefficients_low_level works only with linear constraints in the cplex interface.')

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        if value is None:
            # Update expression from solver instance one last time
            self._get_expression()
            self._problem = None
        else:
            self._problem = value

    @property
    def primal(self):
        if self.problem is not None:
            primal_from_solver = self.problem.problem.solution.get_activity_levels(self.name)
            # return self._round_primal_to_bounds(primal_from_solver)  # Test assertions fail
            return primal_from_solver
        else:
            return None

    @property
    def dual(self):
        if self.problem is not None:
            return self.problem.problem.solution.get_dual_values(self.name)
        else:
            return None

    @interface.Constraint.name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        self._name = value
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's name"
                )
            else:
                self.problem.problem.linear_constraints.set_names(old_name, value)
            self.problem.constraints.update_key(old_name)

    @interface.Constraint.lb.setter
    def lb(self, value):
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's bounds"
                )
            if self.ub is not None and value > self.ub:
                raise ValueError(
                    "Lower bound %f is larger than upper bound %f in constraint %s" %
                    (value, self.ub, self)
                )
            sense, rhs, range_value = _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(value, self.ub)
            if self.is_Linear:
                self.problem.problem.linear_constraints.set_rhs(self.name, rhs)
                self.problem.problem.linear_constraints.set_senses(self.name, sense)
                self.problem.problem.linear_constraints.set_range_values(self.name, range_value)
        self._lb = value

    @interface.Constraint.ub.setter
    def ub(self, value):
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's bounds"
                )
            if self.lb is not None and value < self.lb:
                raise ValueError(
                    "Upper bound %f is less than lower bound %f in constraint %s" %
                    (value, self.lb, self)
                )
            sense, rhs, range_value = _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(self.lb, value)
            if self.is_Linear:
                self.problem.problem.linear_constraints.set_rhs(self.name, rhs)
                self.problem.problem.linear_constraints.set_senses(self.name, sense)
                self.problem.problem.linear_constraints.set_range_values(self.name, range_value)
        self._ub = value

    def __iadd__(self, other):
        # if self.problem is not None:
        #     self.problem._add_to_constraint(self.index, other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            super(Constraint, self).__iadd__(other)
            problem_reference._add_constraint(self, sloppy=False)
        else:
            super(Constraint, self).__iadd__(other)
        return self


@six.add_metaclass(inheritdocstring)
class Objective(interface.Objective):
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)

    @property
    def value(self):
        return self.problem.problem.solution.get_objective_value()

    @interface.Objective.direction.setter
    def direction(self, value):
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.objective.set_sense(
                {'min': self.problem.problem.objective.sense.minimize,
                 'max': self.problem.problem.objective.sense.maximize}[value])
        super(Objective, Objective).direction.__set__(self, value)


@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, lp_method='primal', presolve="auto", verbosity=0, timeout=None,
                 solution_target="auto", qp_method="primal", *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.lp_method = lp_method
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout
        self.solution_target = solution_target
        self.qp_method = qp_method

    @property
    def lp_method(self):
        lpmethod = self.problem.problem.parameters.lpmethod
        try:
            value = lpmethod.get()
        except ReferenceError:
            value = lpmethod.default()
        return lpmethod.values[value]

    @lp_method.setter
    def lp_method(self, lp_method):
        if lp_method not in _LP_METHODS:
            raise ValueError("LP Method %s is not valid (choose one of: %s)" % (lp_method, ", ".join(_LP_METHODS)))
        lp_method = getattr(self.problem.problem.parameters.lpmethod.values, lp_method)
        self.problem.problem.parameters.lpmethod.set(lp_method)

    def _set_presolve(self, value):
        if self.problem is not None:
            presolve = self.problem.problem.parameters.preprocessing.presolve
            if value is True:
                presolve.set(presolve.values.on)
            elif value is False or value == "auto":
                presolve.set(presolve.values.off)
            else:
                raise ValueError(str(value) + " is not a valid presolve parameter. Must be True, False or 'auto'.")

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        self._set_presolve(value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):

        class StreamHandler(StringIO):

            def __init__(self, logger, *args, **kwargs):
                StringIO.__init__(self, *args, **kwargs)
                self.logger = logger

        class ErrorStreamHandler(StreamHandler):

            def flush(self):
                self.logger.error(self.getvalue())

        class WarningStreamHandler(StreamHandler):

            def flush(self):
                self.logger.warn(self.getvalue())

        class LogStreamHandler(StreamHandler):

            def flush(self):
                self.logger.debug(self.getvalue())

        class ResultsStreamHandler(StreamHandler):

            def flush(self):
                self.logger.debug(self.getvalue())

        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        error_stream_handler = ErrorStreamHandler(logger)
        warning_stream_handler = WarningStreamHandler(logger)
        log_stream_handler = LogStreamHandler(logger)
        results_stream_handler = LogStreamHandler(logger)
        if self.problem is not None:
            problem = self.problem.problem
            if value == 0:
                problem.set_error_stream(error_stream_handler)
                problem.set_warning_stream(warning_stream_handler)
                problem.set_log_stream(log_stream_handler)
                problem.set_results_stream(results_stream_handler)
            elif value == 1:
                problem.set_error_stream(sys.stderr)
                problem.set_warning_stream(warning_stream_handler)
                problem.set_log_stream(log_stream_handler)
                problem.set_results_stream(results_stream_handler)
            elif value == 2:
                problem.set_error_stream(sys.stderr)
                problem.set_warning_stream(sys.stderr)
                problem.set_log_stream(log_stream_handler)
                problem.set_results_stream(results_stream_handler)
            elif value == 3:
                problem.set_error_stream(sys.stderr)
                problem.set_warning_stream(sys.stderr)
                problem.set_log_stream(sys.stdout)
                problem.set_results_stream(sys.stdout)
            else:
                raise Exception(
                    "%s is not a valid verbosity level ranging between 0 and 3."
                    % value
                )
        self._verbosity = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if self.problem is not None:
            if value is None:
                self.problem.problem.parameters.timelimit.reset()
            else:
                self.problem.problem.parameters.timelimit.set(value)
        self._timeout = value

    @property
    def solution_target(self):
        if self.problem is not None:
            return _SOLUTION_TARGETS[self.problem.problem.parameters.solutiontarget.get()]
        else:
            return None

    @solution_target.setter
    def solution_target(self, value):
        if self.problem is not None:
            if value is None:
                self.problem.problem.parameters.solutiontarget.reset()
            else:
                try:
                    solution_target = _SOLUTION_TARGETS.index(value)
                except ValueError:
                    raise ValueError(
                        "%s is not a valid solution target. Choose between %s" % (value, str(_SOLUTION_TARGETS)))
                self.problem.problem.parameters.solutiontarget.set(solution_target)
        self._solution_target = self.solution_target

    @property
    def qp_method(self):
        value = self.problem.problem.parameters.qpmethod.get()
        return self.problem.problem.parameters.qpmethod.values[value]

    @qp_method.setter
    def qp_method(self, value):
        if value not in _QP_METHODS:
            raise ValueError("%s is not a valid qp_method. Choose between %s" % (value, str(_QP_METHODS)))
        method = getattr(self.problem.problem.parameters.qpmethod.values, value)
        self.problem.problem.parameters.qpmethod.set(method)
        self._qp_method = value


@six.add_metaclass(inheritdocstring)
class Model(interface.Model):
    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        if problem is None:
            self.problem = cplex.Cplex()

        elif isinstance(problem, cplex.Cplex):
            self.problem = problem
            zipped_var_args = zip(self.problem.variables.get_names(),
                                  self.problem.variables.get_lower_bounds(),
                                  self.problem.variables.get_upper_bounds()
                                  )
            for name, lb, ub in zipped_var_args:
                var = Variable(name, lb=lb, ub=ub, problem=self)
                super(Model, self)._add_variables([var])  # This avoids adding the variable to the glpk problem
            zipped_constr_args = zip(self.problem.linear_constraints.get_names(),
                                     self.problem.linear_constraints.get_rows(),
                                     self.problem.linear_constraints.get_senses(),
                                     self.problem.linear_constraints.get_rhs()

                                     )
            variables = self._variables
            for name, row, sense, rhs in zipped_constr_args:
                constraint_variables = [variables[i - 1] for i in row.ind]
                lhs = _unevaluated_Add(*[val * variables[i - 1] for i, val in zip(row.ind, row.val)])
                if isinstance(lhs, int):
                    lhs = sympy.Integer(lhs)
                elif isinstance(lhs, float):
                    lhs = sympy.RealNumber(lhs)
                if sense == 'E':
                    constr = Constraint(lhs, lb=rhs, ub=rhs, name=name, problem=self)
                elif sense == 'G':
                    constr = Constraint(lhs, lb=rhs, name=name, problem=self)
                elif sense == 'L':
                    constr = Constraint(lhs, ub=rhs, name=name, problem=self)
                elif sense == 'R':
                    range_val = self.problem.linear_constraints.get_rhs(name)
                    if range_val > 0:
                        constr = Constraint(lhs, lb=rhs, ub=rhs + range_val, name=name, problem=self)
                    else:
                        constr = Constraint(lhs, lb=rhs + range_val, ub=rhs, name=name, problem=self)
                else:
                    raise Exception('%s is not a recognized constraint sense.' % sense)

                for variable in constraint_variables:
                    try:
                        self._variables_to_constraints_mapping[variable.name].add(name)
                    except KeyError:
                        self._variables_to_constraints_mapping[variable.name] = set([name])

                super(Model, self)._add_constraints(
                    [constr],
                    sloppy=True
                )
            try:
                objective_name = self.problem.objective.get_name()
            except cplex.exceptions.CplexSolverError as e:
                if 'CPLEX Error  1219:' not in str(e):
                    raise e
            else:
                linear_expression = _unevaluated_Add(
                    *[_unevaluated_Mul(sympy.RealNumber(coeff), variables[index]) for index, coeff in
                      enumerate(self.problem.objective.get_linear()) if coeff != 0.])

                try:
                    quadratic = self.problem.objective.get_quadratic()
                except IndexError:
                    quadratic_expression = Zero
                else:
                    quadratic_expression = self._get_quadratic_expression(quadratic)

                self._objective = Objective(
                    linear_expression + quadratic_expression,
                    problem=self,
                    direction=
                    {self.problem.objective.sense.minimize: 'min', self.problem.objective.sense.maximize: 'max'}[
                        self.problem.objective.get_sense()],
                    name=objective_name
                )
        else:
            raise Exception("Provided problem is not a valid CPLEX model.")
        self.configuration = Configuration(problem=self, verbosity=0)

    def __getstate__(self):
        self.update()
        tmp_file = tempfile.mktemp(suffix=".sav")
        self.problem.write(tmp_file)
        cplex_binary = open(tmp_file, 'rb').read()
        repr_dict = {'cplex_binary': cplex_binary, 'status': self.status, 'config': self.configuration}
        return repr_dict

    def __setstate__(self, repr_dict):
        tmp_file = tempfile.mktemp(suffix=".sav")
        open(tmp_file, 'wb').write(repr_dict['cplex_binary'])
        problem = cplex.Cplex(tmp_file)
        if repr_dict['status'] == 'optimal':
            # turn off logging completely, get's configured later
            problem.set_error_stream(None)
            problem.set_warning_stream(None)
            problem.set_log_stream(None)
            problem.set_results_stream(None)
            problem.solve()  # since the start is an optimal solution, nothing will happen here
        self.__init__(problem=problem)
        self.configuration = Configuration.clone(repr_dict['config'], problem=self)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        if self._objective is not None:
            for variable in self.objective.variables:
                try:
                    self.problem.objective.set_linear(variable.name, 0.)
                except cplex.exceptions.CplexSolverError as e:
                    if " 1210:" not in str(e):  # 1210 = Name not found (variable has been removed from model)
                        raise e
            if self.objective.is_Quadratic:
                if self._objective.expression.is_Mul:
                    args = (self._objective.expression,)
                else:
                    args = self._objective.expression.args
                for arg in args:
                    vars = tuple(arg.atoms(sympy.Symbol))
                    assert len(vars) <= 2
                    try:
                        if len(vars) == 1:
                            self.problem.objective.set_quadratic_coefficients(vars[0].name, vars[0].name, 0)
                        else:
                            self.problem.objective.set_quadratic_coefficients(vars[0].name, vars[1].name, 0)
                    except cplex.exceptions.CplexSolverError as e:
                        if " 1210:" not in str(e):  # 1210 = Name not found (variable has been removed from model)
                            raise e

        super(Model, self.__class__).objective.fset(self, value)
        expression = self._objective.expression
        if isinstance(expression, float) or isinstance(expression, int) or expression.is_Number:
            pass
        else:
            if expression.is_Symbol:
                self.problem.objective.set_linear(expression.name, 1.)
            if expression.is_Mul:
                terms = (expression,)
            elif expression.is_Add:
                terms = expression.args
            else:
                raise ValueError(
                    "Provided objective %s doesn't seem to be appropriate." %
                    self._objective)

            for term in terms:
                factors = term.args
                coeff = factors[0]
                vars = factors[1:]
                assert len(vars) <= 2
                if len(vars) == 2:
                    if vars[0].name == vars[1].name:
                        self.problem.objective.set_quadratic_coefficients(vars[0].name, vars[1].name, 2 * float(coeff))
                    else:
                        self.problem.objective.set_quadratic_coefficients(vars[0].name, vars[1].name, float(coeff))
                else:
                    if vars[0].is_Symbol:
                        self.problem.objective.set_linear(vars[0].name, float(coeff))
                    elif vars[0].is_Pow:
                        var = vars[0].args[0]
                        self.problem.objective.set_quadratic_coefficients(var.name, var.name, 2 * float(
                            coeff))  # Multiply by 2 because it's on diagonal

            self.problem.objective.set_sense(
                {'min': self.problem.objective.sense.minimize, 'max': self.problem.objective.sense.maximize}[
                    value.direction])
        self.problem.objective.set_name(value.name)
        value.problem = self

    @property
    def primal_values(self):
        if self.problem:
            primal_values = collections.OrderedDict()
            for variable, primal in zip(self.variables, self.problem.solution.get_values()):
                primal_values[variable.name] = variable._round_primal_to_bounds(primal)
            return primal_values
        else:
            return None

    @property
    def reduced_costs(self):
        if self.problem:
            return collections.OrderedDict(
                zip([variable.name for variable in self.variables], self.problem.solution.get_reduced_costs()))
        else:
            return None

    @property
    def dual_values(self):
        if self.problem:
            return collections.OrderedDict(
                zip([constraint.name for constraint in self.constraints], self.problem.solution.get_activity_levels()))
        else:
            return None

    @property
    def shadow_prices(self):
        if self.problem:
            return collections.OrderedDict(
                zip([constraint.name for constraint in self.constraints], self.problem.solution.get_dual_values()))
        else:
            return None

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        self.problem.write(tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def _optimize(self):
        self.problem.solve()
        cplex_status = self.problem.solution.get_status()
        self._original_status = self.problem.solution.get_status_string()
        self._status = _CPLEX_STATUS_TO_STATUS[cplex_status]
        return self.status

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        lb = [(variable.name, value) for variable, value in var_lb]
        if len(lb) > 0:
            self.problem.variables.set_lower_bounds(lb)
        ub = [(variable.name, value) for variable, value in var_ub]
        if len(ub) > 0:
            self.problem.variables.set_upper_bounds(ub)

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        lb, ub, vtype, names, coeff = list(), list(), list(), list(), list()
        for variable in variables:
            if variable.lb is None:
                lb.append(-cplex.infinity)
            else:
                lb.append(variable.lb)
            if variable.ub is None:
                ub.append(cplex.infinity)
            else:
                ub.append(variable.ub)
            vtype.append(_VTYPE_TO_CPLEX_VTYPE[variable.type])
            names.append(variable.name)
            coeff.append(0.)
            variable.problem = self

        if set(vtype) == set(['C']):  # this is needed because CPLEX will automatically set the problem_type to MILP if types are specified
            self.problem.variables.add(coeff, lb=lb, ub=ub, names=names)
        else:
            self.problem.variables.add(coeff, lb=lb, ub=ub, types=vtype, names=names)

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal from sympy expressions
        if self.objective is not None:
            self.objective._expression = self.objective.expression.xreplace({var: 0 for var in variables})
        self.problem.variables.delete([variable.name for variable in variables])
        for variable in variables:
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)

        for constraint in constraints:
            constraint._problem = None
            linear_constraints = dict(lin_expr=[], senses=[], rhs=[], range_values=[], names=[])
            if constraint.is_Linear:
                if constraint.expression.is_Add:
                    coeff_dict = constraint.expression.as_coefficients_dict()
                    indices = [var.name for var in coeff_dict.keys()]
                    values = [float(val) for val in coeff_dict.values()]
                elif constraint.expression.is_Mul:
                    variable = list(constraint.expression.atoms(sympy.Symbol))[0]
                    indices = [variable.name]
                    values = [float(constraint.expression.coeff(variable))]
                elif constraint.expression.is_Atom and constraint.expression.is_Symbol:
                    indices = [constraint.expression.name]
                    values = [1.]
                elif constraint.expression.is_Number:
                    indices = []
                    values = []
                else:
                    raise ValueError('Something is fishy with constraint %s' % constraint)

                sense, rhs, range_value = _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(constraint.lb,
                                                                                                   constraint.ub)
                if constraint.indicator_variable is None:
                    linear_constraints['lin_expr'].append(cplex.SparsePair(ind=indices, val=values))
                    linear_constraints['senses'].append(sense)
                    linear_constraints['rhs'].append(rhs)
                    linear_constraints['range_values'].append(range_value)
                    linear_constraints['names'].append(constraint.name)
                else:
                    if sense == 'R':
                        raise ValueError(
                            'CPLEX does not support indicator constraints that have both an upper and lower bound.')
                    else:
                        # Indicator constraints cannot be added in batch
                        self.problem.indicator_constraints.add(
                            lin_expr=cplex.SparsePair(ind=indices, val=values), sense=sense, rhs=rhs,
                            name=constraint.name,
                            indvar=constraint.indicator_variable.name, complemented=abs(constraint.active_when) - 1)
            elif constraint.is_Quadratic:
                raise NotImplementedError('Quadratic constraints (like %s) are not supported yet.' % constraint)
            else:
                raise ValueError(
                    "CPLEX only supports linear or quadratic constraints. %s is neither linear nor quadratic." % constraint)
            self.problem.linear_constraints.add(**linear_constraints)
            constraint.problem = self

    def _remove_constraints(self, constraints):
        super(Model, self)._remove_constraints(constraints)
        for constraint in constraints:
            if constraint.is_Linear:
                self.problem.linear_constraints.delete(constraint.name)
            elif constraint.is_Quadratic:
                self.problem.quadratic_constraints.delete(constraint.name)

    def _set_linear_objective_term(self, variable, coefficient):
        self.problem.objective.set_linear(variable.name, float(coefficient))

    def _get_quadratic_expression(self, quadratic=None):
        if quadratic is None:
            try:
                quadratic = self.problem.objective.get_quadratic()
            except IndexError:
                return Zero
        terms = []
        for i, sparse_pair in enumerate(quadratic):
            for j, val in zip(sparse_pair.ind, sparse_pair.val):
                if i < j:
                    terms.append(val * self._variables[i] * self._variables[j])
                elif i == j:
                    terms.append(0.5 * val * self._variables[i] ** 2)
                else:
                    pass  # Only look at upper triangle
        return _unevaluated_Add(*terms)
