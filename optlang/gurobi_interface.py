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


from warnings import warn

warn("Be careful! The GUROBI interface is still under construction ...")

import logging

log = logging.getLogger(__name__)

import six
import tempfile
from optlang import interface
from optlang.util import inheritdocstring
import sympy
from sympy.core.add import _unevaluated_Add

import gurobipy

_GUROBI_STATUS_TO_STATUS = {
    gurobipy.GRB.LOADED: interface.LOADED,
    gurobipy.GRB.OPTIMAL: interface.OPTIMAL,
    gurobipy.GRB.INFEASIBLE: interface.INFEASIBLE,
    gurobipy.GRB.INF_OR_UNBD: interface.INFEASIBLE_OR_UNBOUNDED,
    gurobipy.GRB.UNBOUNDED: interface.UNBOUNDED,
    gurobipy.GRB.CUTOFF: interface.CUTOFF,
    gurobipy.GRB.ITERATION_LIMIT: interface.ITERATION_LIMIT,
    gurobipy.GRB.NODE_LIMIT: interface.NODE_LIMIT,
    gurobipy.GRB.TIME_LIMIT: interface.TIME_LIMIT,
    gurobipy.GRB.SOLUTION_LIMIT: interface.SOLUTION_LIMIT,
    gurobipy.GRB.INTERRUPTED: interface.INTERRUPTED,
    gurobipy.GRB.NUMERIC: interface.NUMERIC,
    gurobipy.GRB.SUBOPTIMAL: interface.SUBOPTIMAL,
    gurobipy.GRB.INPROGRESS: interface.INPROGRESS
}

_VTYPE_TO_GUROBI_VTYPE = {'continuous': gurobipy.GRB.CONTINUOUS, 'integer': gurobipy.GRB.INTEGER,
                          'binary': gurobipy.GRB.BINARY}
_GUROBI_VTYPE_TO_VTYPE = dict((val, key) for key, val in _VTYPE_TO_GUROBI_VTYPE.items())


def _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(lb, ub):
    """Helper function used by Constraint and Model"""
    if lb is None and ub is None:
        raise Exception("Free constraint ...")
    elif lb is None:
        sense = '<'
        rhs = float(ub)
        range_value = 0.
    elif ub is None:
        sense = '>'
        rhs = float(lb)
        range_value = 0.
    elif lb == ub:
        sense = '='
        rhs = float(lb)
        range_value = 0.
    elif lb > ub:
        raise ValueError("Lower bound is larger than upper bound.")
    else:
        sense = '='
        rhs = float(lb)
        range_value = float(ub - lb)
    return sense, rhs, range_value


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)
        self._original_name = name  # due to bug with getVarByName ... TODO: file bug report on gurobi google group

    @property
    def _internal_variable(self):
        if self.problem:

            internal_variable = self.problem.problem.getVarByName(self._original_name)

            assert internal_variable is not None
            # if internal_variable is None:
            #     raise Exception('Cannot find variable {}')
            return internal_variable
        else:
            return None

    @interface.Variable.lb.setter
    def lb(self, value):
        super(Variable, self.__class__).lb.fset(self, value)
        if self.problem:
            return self._internal_variable.setAttr('LB', value)

    @interface.Variable.ub.setter
    def ub(self, value):
        super(Variable, self.__class__).ub.fset(self, value)
        if self.problem:
            return self._internal_variable.setAttr('UB', value)

    @interface.Variable.type.setter
    def type(self, value):
        if self.problem:
            return self._internal_variable.setAttr('VType', _VTYPE_TO_GUROBI_VTYPE[value])
        super(Variable, Variable).type.fset(self, value)

    @property
    def primal(self):
        if self.problem:
            return self._internal_variable.getAttr('X')
        else:
            return None

    @property
    def dual(self):
        if self.problem:
            return self._internal_variable.getAttr('RC')
        else:
            return None

    @interface.Variable.name.setter
    def name(self, value):
        if getattr(self, 'problem', None) is not None:
            # raise NotImplementedError('Cannot change variable names due to a bug in gurobipy.')
            self._internal_variable.setAttr('VarName', value)
            self.problem.problem.update()
        self._name = value


@six.add_metaclass(inheritdocstring)
class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)
        if self.ub is not None and self.lb is not None and self.lb > self.ub:
            raise ValueError(
                "Lower bound %f is larger than upper bound %f in constraint %s" %
                (self.lb, self.ub, self)
            )

    @property
    def _internal_constraint(self):
        if self.problem is not None:
            return self.problem.problem.getConstrByName(self.name)

    def _get_expression(self):
        if self.problem is not None:
            gurobi_problem = self.problem.problem
            gurobi_constraint = self._internal_constraint
            row = gurobi_problem.getRow(gurobi_constraint)
            terms = []
            for i in range(row.size()):
                internal_var_name = row.getVar(i).VarName
                if internal_var_name == self.name + '_aux':
                    continue
                variable = self.problem._variables[internal_var_name]
                coeff = sympy.RealNumber(row.getCoeff(i))
                terms.append(sympy.Mul._from_args((coeff, variable)))
            sympy.Add._from_args(terms)
            self._expression = sympy.Add._from_args(terms)
        return self._expression

    def _set_coefficients_low_level(self, variables_coefficients_dict):
        raise NotImplemented

    def __str__(self):
        if self.problem is not None:
            self.problem.problem.update()
        return super(Constraint, self).__str__()

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
            return self._internal_constraint.Slack
            # return self._round_primal_to_bounds(primal_from_solver)  # Test assertions fail
            # return primal_from_solver
        else:
            return None

    @property
    def dual(self):
        if self.problem is not None:
            return self._internal_constraint.Pi
        else:
            return None

    @interface.Constraint.name.setter
    def name(self, value):
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's name"
                )
            else:
                # TODO: the following needs to deal with quadratic constraints
                self.problem.problem.linear_constraints.set_names(self.name, value)
        self._name = value

    @interface.Constraint.lb.setter
    def lb(self, value):
        if getattr(self, 'problem', None) is not None:
            if self.ub is not None and value > self.ub:
                raise ValueError(
                    "Lower bound %f is larger than upper bound %f in constraint %s" %
                    (value, self.ub, self)
                )
            gurobi_constraint = self.problem.problem.getConstrByName(self.name)
            sense, rhs, range_value = _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(value, self.ub)
            if range_value == 0.:
                gurobi_constraint.setAttr('Sense', sense)
                gurobi_constraint.setAttr('RHS', rhs)
            else:
                aux_var = self.problem.problem.getVarByName(gurobi_constraint.name + '_aux')
                if aux_var is None:
                    aux_var = self.problem.problem.addVar(name=gurobi_constraint.name + '_aux', lb=0, ub=range_value)
                    row = self.problem.problem.getRow(gurobi_constraint)
                    updated_row = row - aux_var
                    self.problem.problem.remove(gurobi_constraint)
                    self.problem.problem.update()
                    self.problem.problem.addConstr(updated_row, sense, rhs, self.name)
                else:
                    aux_var.setAttr("UB", range_value)
        self._lb = value

    @interface.Constraint.ub.setter
    def ub(self, value):
        if getattr(self, 'problem', None) is not None:
            if self.lb is not None and value < self.lb:
                raise ValueError(
                    "Upper bound %f is less than lower bound %f in constraint %s" %
                    (value, self.lb, self)
                )
            gurobi_constraint = self.problem.problem.getConstrByName(self.name)
            sense, rhs, range_value = _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(self.lb, value)
            if range_value == 0.:
                gurobi_constraint.setAttr('Sense', sense)
                gurobi_constraint.setAttr('RHS', rhs)
            else:
                aux_var = self.problem.problem.getVarByName(gurobi_constraint.name + '_aux')
                if aux_var is None:
                    aux_var = self.problem.problem.addVar(name=gurobi_constraint.name + '_aux', lb=0, ub=range_value)
                    row = self.problem.problem.getRow(gurobi_constraint)
                    updated_row = row - aux_var
                    self.problem.problem.remove(gurobi_constraint)
                    self.problem.problem.update()
                    self.problem.problem.addConstr(updated_row, sense, rhs, self.name)
                else:
                    aux_var.setAttr("UB", range_value)
        self._ub = value

    def __iadd__(self, other):
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
        try:
            return self.problem.problem.getAttr("ObjVal")
        except gurobipy.GurobiError:  # TODO: let this check the actual error message
            return None

    @interface.Objective.direction.setter
    def direction(self, value):
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.setAttr('ModelSense', {'min': 1, 'max': -1}[self.problem.objective.direction])
        super(Objective, Objective).direction.__set__(self, value)


@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, lp_method='primal', tolerance=1e-9, presolve=False, verbosity=0, timeout=None,
                 solution_target="auto", qp_method="primal", *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.lp_method = lp_method
        self.tolerance = tolerance
        self.presolve = presolve
        self._verbosity = verbosity
        self.timeout = timeout
        self.solution_target = solution_target
        self.qp_method = qp_method

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        if value is True:
            self.problem.problem.params.Presolve = 2
        elif value is False:
            self.problem.problem.params.Presolve = 0
        else:
            self.problem.problem.params.Presolve = -1
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if self.problem is not None:
            if value == 0:
                self.problem.problem.params.OutputFlag = 0
            elif 0 < value <= 3:
                self.problem.problem.params.OutputFlag = 1
        self._verbosity = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if self.problem is not None:
            if value is None:
                self.problem.problem.params.TimeLimit = gurobipy.GRB.INFINITY
            else:
                self.problem.problem.params.TimeLimit = value
        self._timeout = value


class Model(interface.Model):
    """docstring for Model"""

    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        if problem is None:
            self.problem = gurobipy.Model()
            if self.name is not None:
                self.problem.setAttr('ModelName', self.name)
                self.problem.update()

        elif isinstance(problem, gurobipy.Model):
            self.problem = problem
            variables = []
            for gurobi_variable in self.problem.getVars():
                variables.append(Variable(
                    gurobi_variable.getAttr("VarName"),
                    lb=gurobi_variable.getAttr("lb"),
                    ub=gurobi_variable.getAttr("ub"),
                    problem=self,
                    type=_GUROBI_VTYPE_TO_VTYPE[gurobi_variable.getAttr("vType")]
                ))
            super(Model, self)._add_variables(variables)
            constraints = []
            for gurobi_constraint in self.problem.getConstrs():
                sense = gurobi_constraint.Sense
                name = gurobi_constraint.getAttr("ConstrName")
                rhs = gurobi_constraint.RHS
                row = self.problem.getRow(gurobi_constraint)
                lhs = _unevaluated_Add(
                    *[sympy.RealNumber(row.getCoeff(i)) * self.variables[row.getVar(i).VarName] for i in
                      range(row.size())])

                if sense == '=':
                    constraint = Constraint(lhs, name=name, lb=rhs, ub=rhs, problem=self)
                elif sense == '>':
                    constraint = Constraint(lhs, name=name, lb=rhs, ub=None, problem=self)
                elif sense == '<':
                    constraint = Constraint(lhs, name=name, lb=None, ub=rhs, problem=self)
                else:
                    raise ValueError('{} is not a valid sense'.format(sense))
                constraints.append(constraint)
            super(Model, self)._add_constraints(constraints, sloppy=True)

            gurobi_objective = self.problem.getObjective()
            linear_expression = _unevaluated_Add(
                *[sympy.RealNumber(gurobi_objective.getCoeff(i)) * self.variables[gurobi_objective.getVar(i).VarName]
                  for i in range(gurobi_objective.size())])

            self._objective = Objective(
                linear_expression,
                problem=self,
                direction={1: 'min', -1: 'max'}[self.problem.getAttr('ModelSense')]
            )
        else:
            raise TypeError("Provided problem is not a valid Gurobi model.")

        self.configuration = Configuration(problem=self, verbosity=0)

    def __getstate__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        self.problem.write(tmp_file)
        lp = open(tmp_file, 'rb').read()
        repr_dict = {'lp': lp, 'status': self.status, 'config': self.configuration}
        return repr_dict

    def __setstate__(self, repr_dict):
        tmp_file = tempfile.mktemp(suffix=".lp")
        open(tmp_file, 'wb').write(repr_dict['lp'])
        problem = gurobipy.read(tmp_file)
        # if repr_dict['status'] == 'optimal':  # TODO: uncomment this
        #     # turn off logging completely, get's configured later
        #     problem.set_error_stream(None)
        #     problem.set_warning_stream(None)
        #     problem.set_log_stream(None)
        #     problem.set_results_stream(None)
        #     problem.solve()  # since the start is an optimal solution, nothing will happen here
        self.__init__(problem=problem)
        # self.configuration = Configuration.clone(repr_dict['config'], problem=self)  # TODO: make configuration work

    def __str__(self):
        self.problem.update()
        tmp_file = tempfile.mktemp(suffix=".lp")
        self.problem.update()
        self.problem.write(tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def update(self):
        super(Model, self).update(callback=self.problem.update)

    def optimize(self):
        self.problem.optimize()
        self._status = _GUROBI_STATUS_TO_STATUS[self.problem.getAttr("Status")]
        return self.status

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        for variable in variables:
            if variable.lb is None:
                lb = -gurobipy.GRB.INFINITY
            else:
                lb = variable.lb
            if variable.ub is None:
                ub = gurobipy.GRB.INFINITY
            else:
                ub = variable.ub
            gurobi_var = self.problem.addVar(
                name=variable.name,
                lb=lb,
                ub=ub,
                vtype=_VTYPE_TO_GUROBI_VTYPE[variable.type]
            )
            variable._internal_var = gurobi_var

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal from sympy expressions
        self.objective._expression = self.objective.expression.xreplace({var: 0 for var in variables})
        for variable in variables:
            internal_variable = variable._internal_variable
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]
            self.problem.remove(internal_variable)

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)
        for constraint in constraints:
            if constraint.lb is None and constraint.ub is None:
                raise ValueError("optlang does not support free constraints in the gurobi interface.")
            self.problem.update()
            constraint._problem = None
            if constraint.is_Linear:
                if constraint.expression.is_Add:
                    coeff_dict = constraint.expression.as_coefficients_dict()
                    variables = [self.problem.getVarByName(variable.name) for variable in coeff_dict.keys()]
                    coefficients = [float(val) for val in coeff_dict.values()]
                elif constraint.expression.is_Mul:
                    variable = list(constraint.expression.atoms(sympy.Symbol))[0]
                    variables = [self.problem.getVarByName(variable.name)]
                    coefficients = [float(constraint.expression.coeff(variable))]
                elif constraint.expression.is_Atom and constraint.expression.is_Symbol:
                    variables = [self.problem.getVarByName(constraint.expression.name)]
                    coefficients = [1.]
                elif constraint.expression.is_Number:
                    variables = []
                    coefficients = []
                else:
                    raise ValueError('Something is fishy with constraint %s' % constraint)
                lhs = gurobipy.quicksum([coeff * var for coeff, var in zip(coefficients, variables)])
                sense, rhs, range_value = _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(constraint.lb,
                                                                                                    constraint.ub)
                if range_value != 0.:
                    aux_var = self.problem.addVar(name=constraint.name + '_aux', lb=0, ub=range_value)
                    self.problem.update()
                    lhs = lhs - aux_var
                    rhs = constraint.ub
                self.problem.addConstr(lhs, sense, rhs, name=constraint.name)
            else:
                raise ValueError(
                    "GUROBI currently only supports linear constraints. %s is not linear." % self)
                # self.problem.addQConstr(lhs, sense, rhs)
            constraint.problem = self

    def _remove_constraints(self, constraints):
        self.problem.update()
        internal_constraints = [constraint._internal_constraint for constraint in constraints]
        super(Model, self)._remove_constraints(constraints)
        for internal_constraint in internal_constraints:
            self.problem.remove(internal_constraint)


if __name__ == '__main__':
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    z = Variable('z', lb=-100, ub=99.)
    constr = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')

    solver = Model(problem=gurobipy.read("tests/data/model.lp"))

    solver.add(z)
    solver.add(constr)
    print(solver)
    print(solver.optimize())
    print(solver.status)
