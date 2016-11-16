# Copyright 2015 Novo Nordisk Foundation Center for Biosustainability,
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


"""Interface for GAMS"""

import logging

import gams
import six
from optlang import interface

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# TODO: invalidate any results if model changes

# https://www.gams.com/help/index.jsp?topic=%2Fgams.doc%2Fapis%2Fpython%2Findex.html
_GAMS_STATUS_TO_STATUS = {
    gams.ModelStat.Feasible: interface.FEASIBLE,
    gams.ModelStat.InfeasibleLocal: interface.INFEASIBLE,
    gams.ModelStat.InfeasibleGlobal: interface.INFEASIBLE,
    gams.ModelStat.OptimalGlobal: interface.OPTIMAL,
    gams.ModelStat.Unbounded: interface.UNBOUNDED,
    gams.ModelStat.UnboundedNoSolution: interface.UNBOUNDED
}


class Variable(interface.Variable):
    def __init__(self, *args, **kwargs):
        super(Variable, self).__init__(*args, **kwargs)

    @property
    def primal(self):
        if self.problem is None:
            return None
        else:
            if self.problem._column_primals is None:
                return None
            else:
                return self.problem._column_primals[self.name]

    @property
    def dual(self):
        if self.problem is None:
            return None
        else:
            if self.problem._column_duals is None:
                return None
            else:
                return self.problem._column_duals[self.name]


class Constraint(interface.Constraint):
    def __init__(self, *args, **kwargs):
        super(Constraint, self).__init__(*args, **kwargs)

    @property
    def primal(self):
        if self.problem is None:
            return None
        else:
            if self.problem._row_primals is None:
                return None
            else:
                return self.problem._row_primals[self.name]

    @property
    def dual(self):
        if self.problem is None:
            return None
        else:
            if self.problem._row_duals is None:
                return None
            else:
                return self.problem._row_duals[self.name]


class Objective(interface.Objective):
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)

    @property
    def value(self):
        if not hasattr(self.problem, '_out_db', None):
            return self.problem._out_db['Z'].value
        else:
            return None


class Configuration(interface.MathematicalProgrammingConfiguration):
    """docstring for Configuration"""

    def __init__(self, presolve=False, verbosity=0, timeout=None, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)


class Model(interface.Model):
    """GLPK solver interface"""

    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)
        self.configuration = Configuration()
        self._column_primals = None
        self._column_duals = None
        self._row_primals = None
        self._row_duals = None

    @property
    def primal_values(self):
        return self._column_primals

    @property
    def reduced_costs(self):
        return self._column_duals

    @property
    def dual_values(self):
        return self._row_primals

    @property
    def shadow_prices(self):
        return self._row_duals

    def __to_gams(self):

        def constraints2gams(constraints):
            gams_constraints = ''
            constraint_names = list()
            for constraint in constraints:
                string = '{}.. {}'.format(constraint.name, constraint.expression)
                lb, ub = constraint.lb, constraint.ub
                if lb is None and ub is None:
                    return ''
                elif lb is None:
                    constraint_names.append(constraint.name)
                    gams_constraint = '{} =L= {};\n'.format(string, ub)
                elif ub is None:
                    constraint_names.append(constraint.name)
                    gams_constraint = '{} =G= {};\n'.format(string, lb)
                elif ub == lb:
                    constraint_names.append(constraint.name)
                    gams_constraint = '{} =E= {};\n'.format(string, lb)
                else:
                    constraint_names.append('LB_' + constraint.name)
                    constraint_names.append('UB_' + constraint.name)
                    gams_constraint = 'LB_{} =G= {};\nUB_{} =L= {};\n'.format(string, lb, string, ub)
                gams_constraints += gams_constraint
            return gams_constraints, constraint_names

        def lb2gams(bound):
            if bound is None:
                return '-INF'
            else:
                return bound

        def ub2gams(bound):
            if bound is None:
                return 'INF'
            else:
                return bound

        variables = 'Variables Z, {};\n'.format(', '.join(variable.name for variable in self.variables))

        # constraints
        equations, constraint_names = constraints2gams(self.constraints)
        equations_ids = 'Equations obj, {};\n'.format(', '.join(constraint_names))
        # objective
        objective = '\n* objective\nobj.. Z =E= {};'.format(str(self.objective.expression))
        # bounds
        lower_bounds = '\n'.join(
            variable.name + '.lo = {};'.format(lb2gams(variable.lb)) for variable in self.variables)
        upper_bounds = '\n'.join(
            variable.name + '.up = {};'.format(ub2gams(variable.ub)) for variable in self.variables)
        gams_template = '''{variables}
{equations_ids}
{equations}
{objective}

* lower bounds
{lower_bounds}

* upper_bounds
{upper_bounds}

Model problem / ALL /;
Solve problem {direction} Z USING {problem_type};
'''
        return gams_template.format(variables=variables, equations_ids=equations_ids, equations=equations,
                                    objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                                    direction={'min': 'Minimizing', 'max': 'Maximizing'}[self.objective.direction],
                                    problem_type=self.__determine_problem_type())

    def __determine_problem_type(self):
        # TODO: make this functional.
        return 'LP'

    def __str__(self):
        return self.__to_gams()

    def _optimize(self):
        workspace = gams.GamsWorkspace(system_directory='/Applications/GAMS24.7/sysdir', debug=3)
        problem_string = self.__str__() + "\nScalar MODELSTATUS;\n MODELSTATUS = problem.modelstat;"
        job = workspace.add_job_from_string(problem_string)
        job.run()
        self._out_db = job.out_db
        self._column_primals = {variable.name: job.out_db[variable.name].find_record().level for variable in
                                self.variables}
        self._column_duals = {variable.name: job.out_db[variable.name].find_record().marginal for variable in
                              self.variables}
        # need to deal with range constraints here ...
        # self._row_primals = {constraint: job.out_db.get_symbol(constraint.name).find_record().level for constraint in self.constraints}
        # self._row_duals = {constraint: job.out_db.get_symbol(constraint.name).find_record().marginal for constraint in
        #                      self.constraints}
        print(problem_string)
        self._original_status = int(job.out_db['MODELSTATUS'].find_record().value)
        self._status = _GAMS_STATUS_TO_STATUS[self._original_status]
        return self.status


if __name__ == '__main__':
    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)
    c1 = Constraint(x1 + x2 + x3, lb=-100, ub=100, name='c1')
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])
    status = model.optimize()
    print("status:", model.status)
    print("objective value:", model.objective.value)
    print(model.variables)
    for var_name, var in six.iteritems(model.variables):
        print(var_name, "=", var.primal)
