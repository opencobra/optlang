# Copyright 2017 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The scipy interface uses the 'linprog' function of the scipy package.
It has no integer or QP capabilities. Furthermore dual values (reduced costs and shadow prices) are
not supported by this interface.

This interface works well with small to medium scale models but for better performance, other
solvers should be used for large models.
"""

from __future__ import absolute_import, print_function

import tempfile
from collections import OrderedDict
from itertools import islice

import re
import six
import subprocess

import sys

from optlang import interface
from optlang.util import inheritdocstring


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, lb=None, ub=None, type="continuous", *args, **kwargs):
        if type != "continuous":
            raise ValueError("soplex only works with continuous variables. Please use another interface")
        super(Variable, self).__init__(name, lb, ub, type, *args, **kwargs)

    def _get_primal(self):
        return self.problem._primals[self]


@six.add_metaclass(inheritdocstring)
class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Constraint, self).__init__(expression, sloppy=sloppy, *args, **kwargs)
        if not sloppy:
            if not self.is_Linear:
                raise ValueError(
                    "soplex only supports linear constraints. %s is not linear." % self)


@six.add_metaclass(inheritdocstring)
class Objective(interface.Objective):
    def __init__(self, expression, sloppy=False, **kwargs):
        super(Objective, self).__init__(expression, sloppy=sloppy, **kwargs)
        if not (sloppy or self.is_Linear):
            raise ValueError(
                "soplex only supports linear objectives. %s is not linear." % self)


@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, verbosity=0, tolerance=1e-9, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self._verbosity = verbosity
        self.tolerance = tolerance

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = value

    @property
    def presolve(self):
        return False

    @presolve.setter
    def presolve(self, value):
        if value is not False:
            raise ValueError("The Scipy solver has no presolve capabilities")

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    @property
    def timeout(self):
        return self._timeout  # TODO: use timeout in optimize

    @verbosity.setter
    def timeout(self, value):
        self._timeout = value


@six.add_metaclass(inheritdocstring)
class Model(interface.Model):
    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        # if problem is None:
        #     self.problem = Problem()
        # else:
        #     if isinstance(problem, Problem):
        #         self.problem = problem
        #     else:
        #         raise TypeError("Problem must be an instance of scipy_interface.Problem, not " + repr(type(problem)))

        self.configuration = Configuration(problem=self)

    def to_lp(self):
        from optlang import glpk_interface
        return str(glpk_interface.Model.clone(self))

    @staticmethod
    def _parse_objective_value(cli_output):
        regex = 'Objective value     : ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
        return float(re.findall(regex, cli_output)[0][0])

    @staticmethod
    def _parse_primals(cli_output):
        regex = '(\w+)\t\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
        split_output = cli_output.split('Primal solution (name, value):\n')[1]
        return {id_: float(num) for id_, num, _ in re.findall(regex, split_output)}

    @staticmethod
    def _parse_status(cli_output):
        return re.findall('SoPlex status       : problem is solved \[(\w+)\]', cli_output)[0]

    def _optimize(self):
        with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
            lp = self.__str__().encode('utf-8')
            print(lp.decode('utf-8'))
            f.write(lp)
            f.flush()
            cmd = ['soplex', '-v3', '-x', f.name]
            print(cmd)
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
            print(output)
        self.objective._value = self._parse_objective_value(output)
        parsed_primals = self._parse_primals(output)  # dict contains only non-zero primals
        primals = {}
        for variable_id, variable in self.variables.items():
            if variable_id not in parsed_primals:
                primals[variable] = 0
            else:
                primals[variable] = parsed_primals[variable_id]
        self._primals = primals
        status = self._parse_status(output)
        self._status = status
        return status

    def primal_values(self):
        return self._primals


if __name__ == "__main__":
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

    for var in model.variables:
        print(var.name, "=", var.primal)
