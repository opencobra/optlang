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

import logging
log = logging.getLogger(__name__)
import collections
import sympy


OPTIMAL = 0
UNDEFINED = 1
FEASIBLE = 2
INFEASIBLE = 3
NOFEASIBLE = 4
UNBOUNDED = 5
INFEASIBLE_OR_UNBOUNDED = 6
LOADED = 7
CUTOFF = 8
ITERATION_LIMIT = 9
NODE_LIMIT = 10
TIME_LIMIT = 11
SOLUTION_LIMIT = 12
INTERRUPTED = 13
NUMERIC = 14
SUBOPTIMAL = 15
IN_PROGRESS = 16

# class Status(object):
#     """docstring for Status"""
#     def __init__(self, arg):
#         super(Status, self).__init__()
#         self.arg = arg
        

class Variable(sympy.Symbol):

    """docstring for Column"""

    def __init__(self, name, lb=None, ub=None, type="continuous", problem=None, primal=None, dual=None, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)
        self.lb = lb
        self.ub = ub
        self.type = type
        self.problem = problem
        self.primal = primal
        self.dual = dual

    def __del__(self):
        if self.problem is not None:
            self.problem._remove_variable(self)
            self.problem = None
        del self


class Objective(object):

    """docstring for Objective"""

    def __init__(self, expression, name=None, problem=None, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)
        self.expression = expression
        self.name = name
        self.problem = problem


class Constraint(object):

    """docstring for Constraint"""

    def __init__(self, expression, name=None, lb=None, ub=None, problem=None, *args, **kwargs):
        super(Constraint, self).__init__(*args, **kwargs)
        self.expression = expression
        self.name = name
        self.problem = problem
        self.lb = lb
        self.ub = ub

    def __str__(self):
        if self.lb:
            lhs = str(self.lb) + ' <= '
        else:
            lhs = ''
        if self.ub:
            rhs = '<=' + str(self.lb)
        else:
            rhs = ''
        return str(self.name) + ": " + lhs + self.expression.__str__() + rhs

    def __repr__(self):
        return self.__str__()

    @property
    def variables(self):
        return self.expression.free_symbols


class Solver(object):

    """docstring for Solver"""

    def __init__(self, obj=None, *args, **kwargs):
        super(Solver, self).__init__(*args, **kwargs)
        self.objective = obj
        self.variables = collections.OrderedDict()
        self.constraints = collections.OrderedDict()
        self.status = None

    def __str__(self):
        print self.objective

    def add(self, stuff):
        """Add variables, constraints, ..."""
        if isinstance(stuff, collections.Iterable):
            for elem in stuff:
                self.add(elem)
        elif isinstance(stuff, Variable):
            self._add_variable(stuff)
        elif isinstance(stuff, Constraint):
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
        raise NotImplementedError(
            "You're using the high level interface to optlang. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    def from_cplex(self, cplex_str):
        raise NotImplementedError

    def _add_variable(self, variable):
        variable.problem = self
        self.variables[variable.name] = variable

    def _remove_variable(self, variable):
        if isinstance(variable, Variable):
            var = self.variables[variable.name]
            var.problem = None
            del self.variables[variable.name]
        elif isinstance(variable, str):
            var = self.variables[variable]
            var.problem = None
            del self.variables[variable.name]
        else:
            raise LookupError("Variable %s not in solver" % s)

    def _add_constraint(self, constraint):
        if constraint.name is None:
            self.constraints[constraint.__hash__()] = constraint
        else:
            self.constraints[constraint.name] = constraint

    def _remove_constraint(self, constraint):
        del self.constraints[constraint.__hash__]
        del constraint

class Solution(object):
    """docstring for Solution"""
    def __init__(self, *args, **kwargs):
        super(Solution, self).__init__(*args, **kwargs)
        self.status = None
        self.variable_primals = OrderedDict()
        self.variable_duals = OrderedDict()
        self.constrain_primals = OrderedDict()
        self.variable_primals = OrderedDict()

    def populate(self, solver, var_primals=True, var_duals=True, constr_primals=True, constr_duals=True):
        pass
        

if __name__ == '__main__':
    # Example workflow
    solver = Solver()
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    # constr = Constraint(x + y + z > 3, name="constr1")
    constr = Constraint(x + y, lb=3, name="constr1")
    obj = Objective(2 * x + y)

    solver.add(x)
    solver.add(y)

    solver.add(constr)
    solver.add(obj)

    try:
        sol = solver.optimize()
    except NotImplementedError, e:
        print e
    
    solver.remove(x)
