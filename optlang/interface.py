# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

"""Abstract solver interface definitions (:class:`Model`, :class:`Variable`,
:class:`Constraint`, :class:`Objective`) intended to be subclassed and
extended for individual solvers.
"""

import logging
log = logging.getLogger(__name__)
import collections
import sympy


OPTIMAL = 'optimal'
UNDEFINED = 'undefined'
FEASIBLE = 'feasible'
INFEASIBLE = 'infeasible'
NOFEASIBLE = 'nofeasible'
UNBOUNDED = 'unbounded'
INFEASIBLE_OR_UNBOUNDED = 'infeasible_or_unbouned'
LOADED = 'loaded'
CUTOFF = 'cutoff'
ITERATION_LIMIT = 'iteration_limit'
NODE_LIMIT = 'node_limit'
TIME_LIMIT = 'time_limit'
SOLUTION_LIMIT = 'solution_limit'
INTERRUPTED = 'interrupted'
NUMERIC = 'numeric'
SUBOPTIMAL = 'suboptimal'
INPROGRESS = 16

# class Status(object):
#     """docstring for Status"""
#     def __init__(self, arg):
#         super(Status, self).__init__()
#         self.arg = arg


class Variable(sympy.Symbol):

    """
    Class to model optimization variables. Extends sympy Symbol
    with optimization specific attributes and methods.

    Attributes
    ----------
    name: str
        The variable's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    type: str, optional
        The variable type, 'continuous' or 'integer' or 'binary'.
    problem: Model or None, optional
        A reference to the optimization model the variable belongs to.
    See Also
    --------
    Constraint, Objective, Model

    Examples
    --------
    ...

    """

    def __init__(self, name, lb=None, ub=None, type="continuous", problem=None, *args, **kwargs):
        for char in name:
            if char.isspace():
                raise ValueError(
                    'Variable names cannot contain whitespace characters. "%s" contains whitespace character "%s".' % (name, char))
        super(Variable, self).__init__(name, *args, **kwargs)
        self.lb = lb
        self.ub = ub
        self.type = type  # TODO: binary is probably not a good idea ...
        self.problem = problem
        # self.primal = primal
        # self.dual = dual

    def __str__(self):
        """Print a string representation.

    #     Examples
    #     --------
    #     >>> str(Variable('x', lb=-10, ub=10))
    #     '-10 <= x <= 10'

    #     and

    #     >>> str(Variable('x', lb=-10))
    #     '-10 <= x'

    #     """
        if self.lb is not None:
            lb_str = str(self.lb) + " <= "
        else:
            lb_str = ""
        if self.ub is not None:
            ub_str = " <= " + str(self.ub)
        else:
            ub_str = ""
        return ''.join((lb_str, super(Variable, self).__str__(), ub_str))

    def __repr__(self):
        """Does exactly the same as __str__ for now."""
        return self.__str__()

    def __setattr__(self, name, value):

        if name == 'lb' and hasattr(self, 'ub') and self.ub is not None and value is not None and value > self.ub:
            raise ValueError(
                'The provided lower bound %g is larger than the upper bound %g of variable %s.' % (value, self.ub, self))

        if name == 'ub' and hasattr(self, 'lb') and self.lb is not None and value is not None and value < self.lb:
            raise ValueError(
                'The provided upper bound %g is smaller than the lower bound %g of variable %s.' % (value, self.lb, self))

        elif name == 'type':
            if value in ('continuous', 'integer', 'binary'):
                super(Variable, self).__setattr__(name, value)
            else:
                raise ValueError("'%s' is not a valid variable type. Choose between 'continuous, 'integer', or 'binary'." % value)

        else:
            super(Variable, self).__setattr__(name, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class Constraint(object):

    """
    Class to model optimization constraints. Wraps sympy expressions and extends
    them with optimization specific attributes and methods.


    Attributes
    ----------

    expression: sympy
    name: str, optional
        The constraint's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    problem: Model or None, optional
        A reference to the optimization model the variable belongs to.

    """

    def __init__(self, expression, name=None, lb=None, ub=None, problem=None, *args, **kwargs):
        super(Constraint, self).__init__(*args, **kwargs)
        self.lb = lb
        self.ub = ub
        self._expression = self._canonicalize(expression)
        if name is None:
            self.name = sympy.Dummy().name
        else:
            self.name = name
        self.problem = problem

    def __str__(self):
        if self.lb:
            lhs = str(self.lb) + ' <= '
        else:
            lhs = ''
        if self.ub:
            rhs = ' <= ' + str(self.ub)
        else:
            rhs = ''
        return str(self.name) + ": " + lhs + self.expression.__str__() + rhs

    def _canonicalize(self, expression):
        if expression.is_Atom or expression.is_Mul:
            return expression
        lonely_coeffs = [arg for arg in expression.args if arg.is_Number]
        if lonely_coeffs == []:
            return expression
        assert len(lonely_coeffs) == 1
        coeff = lonely_coeffs[0]
        if self.lb is None and self.ub is None:
            raise ValueError(
                "%s cannot be shaped into canonical form if neither lower or upper constraint bounds are set."
                % expression
                )
        elif self.lb is not None:
            expression = expression - coeff
            self.lb = self.lb - coeff
        else:
            expression = expression - coeff
            self.ub = self.ub - coeff
        return expression

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        self._expression = self._canonicalize(expression)

    @property
    def is_Linear(self):
        try:
            poly = self.expression.as_poly(*self.variables)
        except sympy.PolynomialError:
            poly = None
        if poly is not None:
            return poly.is_linear
        else:
            return False

    @property
    def is_Quadratic(self):
        try:
            poly = self.expression.as_poly(*self.expression.free_symbols)
        except sympy.PolynomialError:
            poly = None
        if poly is not None and poly.is_quadratic and not poly.is_linear:
            return True
        else:
            return False

    @property
    def variables(self):
        return self.expression.free_symbols

    def __iadd__(self, other):
        self.expression += other
        return self

    def __isub__(self, other):
        self.expression -= other
        return self

    def __imul__(self, other):
        self.expression *= other
        return self

    def __idiv__(self, other):
        self.expression /= other
        return self

    def __itruediv__(self, other):
        self.expression /= other
        return self


class Objective(object):

    """docstring for Objective"""

    def __init__(self, expression, name=None, value=None, problem=None, direction='max', *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)
        self._expression = self._canonicalize(expression)
        self.name = name
        self._value = value
        self._direction = direction
        self.problem = problem

    @property
    def value(self):
        return self._value

    def __str__(self):
        return {'max': 'Maximize', 'min': 'Minimize'}[self.direction] + '\n' + str(self.expression)
        # return ' '.join((self.direction, str(self.expression)))

    def _canonicalize(self, expression):
        """Change x + y to 1.*x + 1.*y"""
        expression = 1. * expression
        return expression

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = self._canonicalize(value)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        # TODO: implement direction parsing, e.g. 'Maximize' -> 'max'
        self._direction = value


class Configuration(object):

    def __init__(self):
        pass

    @property
    def presolve(self):
        raise NotImplementedError

    @presolve.setter
    def presolve(self, value):
        raise NotImplementedError

    @property
    def verbose(self):
        raise NotImplementedError

    @verbose.setter
    def verbose(self, value):
        raise NotImplementedError


class Model(object):

    """docstring for Model"""

    def __init__(self, objective=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._objective = objective
        self.variables = collections.OrderedDict()
        self.constraints = collections.OrderedDict()
        self.status = None
        self._presolve = False

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        self._objective = value

    def __str__(self):
        return '\n'.join((
            str(self.objective),
            "subject to",
            '\n'.join([str(constr) for constr in self.constraints.values()]),
            'Bounds',
            '\n'.join([str(var) for var in self.variables.values()])
        ))

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
        if isinstance(stuff, str):
            try:
                variable = self.variables[stuff]
                self._remove_variable(variable)
            except KeyError:
                try:
                    constraint = self.constraints[stuff]
                    self._remove_constraint(constraint)
                except KeyError:
                    raise LookupError(
                        "%s is neither a variable nor a constraint in the current solver instance." % stuff)
        elif isinstance(stuff, collections.Iterable):
            for elem in stuff:
                self.remove(elem)
        elif isinstance(stuff, Variable):
            self._remove_variable(stuff)
        elif isinstance(stuff, Constraint):
            self._remove_constraint(stuff)
        elif isinstance(stuff, Objective):
            raise TypeError(
                "Cannot remove objective %s. Use model.objective = Objective(...) to change the current objective." % stuff)
        else:
            raise TypeError(
                "Cannot remove %s. It neither a variable or constraint." % stuff)

    def optimize(self):
        raise NotImplementedError(
            "You're using the high level interface to optlang. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    def from_cplex(self, cplex_str):
        raise NotImplementedError

    def _add_variable(self, variable):
        variable.problem = self
        self.variables[variable.name] = variable
        return variable

    def _remove_variable(self, variable):
        if isinstance(variable, Variable):
            var = self.variables[variable.name]
            var.problem = None
            del self.variables[variable.name]
        else:
            raise LookupError("Variable %s not in solver" % s)

    def _add_constraint(self, constraint, sloppy=False):
        if sloppy is False:
            for var in constraint.variables:
                if var not in self.variables.values():
                    self._add_variable(var)
        if constraint.name is None:
            self.constraints[constraint.__hash__()] = constraint
        else:
            self.constraints[constraint.name] = constraint

    def _remove_constraint(self, constraint):
        del self.constraints[constraint.__hash__]
        del constraint


# class Solution(object):

#     """docstring for Solution"""

#     def __init__(self, status=None, objval=None, *args, **kwargs):
#         super(Solution, self).__init__(*args, **kwargs)
#         self.status = None
#         self.objval = None
#         self.variables = OrderedDict()
#         self.constraints = OrderedDict()

#     def populate(self):
#         pass

#     @property
#     def variable(self):
#         return self._variable
#     @variable.setter
#     def variable(self, value):
#         self._variable = value


if __name__ == '__main__':
    # Example workflow
    model = Model()
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    # constr = Constraint(x + y + z > 3, name="constr1")
    constr = Constraint(x + y, lb=3, name="constr1")
    obj = Objective(2 * x + y)

    # model.add(x)
    # model.add(y)
    model.add(constr)
    model.add(obj)

    try:
        sol = model.optimize()
    except NotImplementedError, e:
        print e

    print model
    print model.variables

    model.remove(x)
