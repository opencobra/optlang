'''

@author: Nikolaus sonnenschein

   Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
   Technical University of Denmark.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
'''

from collections import Iterable, OrderedDict
import sympy


class Variable(sympy.Symbol):
    """docstring for Column"""
    def __init__(self, name, lb=None, ub=None, obj=0., type=None, problem=None):
        super(Variable, self).__init__(name)
        self.lb = lb
        self.ub = ub
        self.obj = obj
        self.type = type
        self.problem = problem

    def __del__(self):
        if self.problem is not None:
            self.problem.remove(self)
        del self

class Objective(object):
    """docstring for Objective"""
    def __init__(self, expr, problem=None):
        self.expr = expr
        self.problem = problem

class Constraint(object):
    """docstring for Constraint"""
    def __init__(self, expr, name=None):
        self.arg = arg
        
class Solver(object):
    """docstring for Solver"""
    def __init__(self, obj=None):
        self.objective = obj
        self.variables = OrderedDict()
        self.constraints = OrderedDict()

    def __str__(self):
        print self.objective

    def add(self, stuff):
        """Add variables, constraints, ..."""
        if isinstance(stuff, Iterable):
            for elem in stuff:
                self.add(elem)
        elif isinstance(stuff, Variable):
            self._add_variable(stuff)
        elif isinstance(stuff, sympy.Relational):
            self._add_constraint(stuff)
        elif isinstance(stuff, Objective):
            self.objective = stuff
        else:
            raise TypeError("Cannot add %s" % stuff)

    def remove(self, stuff):
        if isinstance(stuff, collections.Iterable):
            for elem in stuff:
                self.remove(elem)
        elif isinstance(stuff, Variable):
            self._remove_variable(stuff)
        elif isinstance(stuff, Constraint):
            self._remove_constraint(stuff)
        else:
            raise TypeError("Cannot remove %s" % stuff)

    def optimize(self):
        raise NotImplementedError

    def _add_variable(self, variable):
        self.variable.problem = self
        self.variables[variable.name] = variable

    def _remove_variable(self, variable):
        if isinstance(variable, Variable):
            del self.variables[variable.name]
        elif isinstance(variable, str):
            del self.variables[variable]
            del variable
        else:
            raise LookupError("Variable %s not in solver" % s)

    def _add_constraint(self, constraint, name=None):
        if name is None:
            self.constraints[constraint.__hash__()] = constraint
        else:
            self.constraints[name] = constraint

    def _remove_constraint(self, constraint):
        del self.constraints[constraint.__hash__]
        del constraint

if __name__ == '__main__':
    # Example workflow
    solver = Solver()
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    constr = Constraint(x + y > 3, name="constr1")
    obj = Objective(2*x + y)
    solver.add(x)
    solver.add(y)
    solver.add(obj)
    sol = solver.optimize()
    print sol


