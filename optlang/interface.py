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


"""Abstract solver interface definitions (:class:`Model`, :class:`Variable`,
:class:`Constraint`, :class:`Objective`) intended to be subclassed and
extended for individual solvers.
"""

import logging
import random
import uuid
import six

import types
import collections

import sys

log = logging.getLogger(__name__)

import sympy
from sympy.core.singleton import S
from sympy.core.logic import fuzzy_bool

from .container import Container

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
MEMORY_LIMIT = 'memory_limit'
NODE_LIMIT = 'node_limit'
TIME_LIMIT = 'time_limit'
SOLUTION_LIMIT = 'solution_limit'
INTERRUPTED = 'interrupted'
NUMERIC = 'numeric'
SUBOPTIMAL = 'suboptimal'
INPROGRESS = 'in_progress'
ABORTED = 'aborted'
SPECIAL = 'check_original_solver_status'



# noinspection PyShadowingBuiltins
class Variable(sympy.Symbol):
    """Optimization variables.

    Extends sympy Symbol with optimization specific attributes and methods.

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

    Examples
    --------
    >>> Variable('x', lb=-10, ub=10)
    '-10 <= x <= 10'
    """

    @classmethod
    def clone(cls, variable, **kwargs):
        return cls(variable.name, lb=variable.lb, ub=variable.ub, type=variable.type, **kwargs)

    def __new__(cls, name, **assumptions):

        if assumptions.get('zero', False):
            return S.Zero
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        if is_commutative is None:
            raise ValueError(
                '''Symbol commutativity must be True or False.''')
        assumptions['commutative'] = is_commutative
        for key in assumptions.keys():
            assumptions[key] = bool(assumptions[key])
        return sympy.Symbol.__xnew__(cls, name, uuid=str(int(round(1e16*random.random()))), **assumptions) # uuid.uuid1()

    def __init__(self, name, lb=None, ub=None, type="continuous", problem=None, *args, **kwargs):
        for char in name:
            if char.isspace():
                raise ValueError(
                    'Variable names cannot contain whitespace characters. "%s" contains whitespace character "%s".' % (
                        name, char))
        sympy.Symbol.__init__(name, *args, **kwargs)  #TODO: change this back to use super
        self._lb = lb
        self._ub = ub
        self._type = type
        self.problem = problem

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, value):
        if hasattr(self, 'ub') and self.ub is not None and value is not None and value > self.ub:
            raise ValueError(
                'The provided lower bound %g is larger than the upper bound %g of variable %s.' % (
                    value, self.ub, self))
        elif self.type == 'integer' and value % 1 != 0.:
            raise ValueError(
                'The provided lower bound %g cannot be assigned to integer variable %s (%g mod 1 != 0).' % (
                    value, self, value))
        else:
            self._lb = value

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, value):
        if hasattr(self, 'lb') and self.lb is not None and value is not None and value < self.lb:
            raise ValueError(
                'The provided upper bound %g is smaller than the lower bound %g of variable %s.' % (
                    value, self.lb, self))
        else:
            self._ub = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if value == 'continuous':
            self._type = value
        elif value == 'integer':
            self._type = value
            try:
                self.lb = round(self.lb)
            except TypeError:
                pass
            try:
                self.ub = round(self.ub)
            except TypeError:
                pass
        elif value == 'binary':
            self._type = value
            self._lb = 0
            self._ub = 1
        else:
            raise ValueError(
                "'%s' is not a valid variable type. Choose between 'continuous, 'integer', or 'binary'." % value)

    def __str__(self):
        """Print a string representation of variable.

        Examples
        --------
        >>> Variable('x', lb=-10, ub=10)
        '-10 <= x <= 10'
        """
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

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __round_primal_to_bounds(self, primal, tolerance=1e-6):
        if (primal >= self.lb or self.lb is None) and (primal <= self.ub or self.ub is None):
                return primal
        else:
            if (primal <= self.lb) and ((self.lb - primal) <= tolerance):
                return self.lb
            elif (primal >= self.ub) and ((self.ub - primal) >= -tolerance):
                return self.ub
            else:
                raise AssertionError('The primal value %s returned by the solver is out of bounds for variable %s (lb=%s, ub=%s)' % (primal, self.name, self.lb, self.ub))


# noinspection PyPep8Naming
class OptimizationExpression(object):
    """Abstract base class for Objective and Constraint."""

    @classmethod
    def _substitute_variables(cls, expression, model=None, **kwargs):
        """Substitutes variables in (optimization)expression (constraint/objective) with variables of the appropriate interface type.
        Attributes
        ----------
        expression: Constraint, Objective
            An optimization expression.
        problem: Model or None, optional
            A reference to an optimization model that should be searched for appropriate variables first.
        """
        interface = sys.modules[cls.__module__]
        variable_substitutions = dict()
        for variable in expression.variables:
            if model is not None and variable.name in model.variables:
                # print variable.name, id(variable.problem)
                variable_substitutions[variable] = model.variables[variable.name]
            else:
                variable_substitutions[variable] = interface.Variable.clone(variable)
        adjusted_expression = expression.expression.xreplace(variable_substitutions)
        return adjusted_expression

    def __init__(self, expression, name=None, problem=None, sloppy=False, *args, **kwargs):
        super(OptimizationExpression, self).__init__(*args, **kwargs)
        if sloppy:
            self._expression = expression
        else:
            self._expression = self._canonicalize(expression)
        if name is None:
            self.name = str(uuid.uuid1())
        else:
            self.name = name
        self._problem = problem

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        self._problem = value

    def _get_expression(self):
        return self._expression

    @property
    def expression(self):
        """The mathematical expression defining the objective/constraint."""
        return self._get_expression()

    @property
    def variables(self):
        """Variables in constraint."""
        return self.expression.atoms(sympy.Symbol)

    def _canonicalize(self, expression):
        if isinstance(expression, float):
            return sympy.RealNumber(expression)
        elif isinstance(expression, int):
            return sympy.Integer(expression)
        else:
            return expression

    @property
    def is_Linear(self):
        """Returns True if constraint is linear (read-only)."""
        return all((len(key.free_symbols)<2 and (key.is_Add or key.is_Mul or key.is_Atom) for key in self.expression.as_coefficients_dict().keys()))

    @property
    def is_Quadratic(self):
        """Returns True if constraint is quadratic (read-only)."""
        try:
            poly = self.expression.as_poly(*self.expression.atoms(sympy.Symbol))
        except sympy.PolynomialError:
            poly = None
        if poly is not None and poly.is_quadratic and not poly.is_linear:
            return True
        else:
            return False

    def __iadd__(self, other):
        self._expression += other
        # self.expression = sympy.Add._from_args((self.expression, other))
        return self

    def __isub__(self, other):
        self._expression -= other
        return self

    def __imul__(self, other):
        self._expression *= other
        return self

    def __idiv__(self, other):
        self._expression /= other
        return self

    def __itruediv__(self, other):
        self._expression /= other
        return self


class Constraint(OptimizationExpression):
    """Optimization constraint.

    Wraps sympy expressions and extends them with optimization specific attributes and methods.

    Attributes
    ----------
    expression: sympy
        The mathematical expression defining the constraint.
    name: str, optional
        The constraint's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    problem: Model or None, optional
        A reference to the optimization model the variable belongs to.
    """

    # @classmethod
    # def clone(cls, constraint, model=None, **kwargs):
    #     return cls(cls._substitute_variables(constraint, model=model), lb=constraint.lb, ub=constraint.ub,
    #                name=constraint.name, problem=constraint.problem, sloppy=True, **kwargs)

    @classmethod
    def clone(cls, constraint, model=None, **kwargs):
        return cls(cls._substitute_variables(constraint, model=model), lb=constraint.lb, ub=constraint.ub,
                   name=constraint.name, sloppy=True, **kwargs)

    def __init__(self, expression, lb=None, ub=None, *args, **kwargs):
        self.lb = lb
        self.ub = ub
        super(Constraint, self).__init__(expression, *args, **kwargs)

    def __str__(self):
        if self.lb is not None:
            lhs = str(self.lb) + ' <= '
        else:
            lhs = ''
        if self.ub is not None:
            rhs = ' <= ' + str(self.ub)
        else:
            rhs = ''
        return str(self.name) + ": " + lhs + self.expression.__str__() + rhs

    def _canonicalize(self, expression):
        expression = super(Constraint, self)._canonicalize(expression)
        if expression.is_Atom or expression.is_Mul:
            return expression
        lonely_coeffs = [arg for arg in expression.args if arg.is_Number]
        if not lonely_coeffs:
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


class Objective(OptimizationExpression):
    """Objective function.

    Attributes
    ----------
    expression: sympy
        The mathematical expression defining the objective.
    name: str, optional
        The name of the constraint.
    direction: 'max' or 'min'
        The optimization direction.
    value: float, read-only
        The current objective value.
    problem: solver
        The low-level solver object.

    """

    @classmethod
    def clone(cls, objective, model=None, **kwargs):
        return cls(cls._substitute_variables(objective, model=model), name=objective.name,  # TODO: problem=model, (it's breaking cameo for some reason)
                   direction=objective.direction, sloppy=True, **kwargs)

    def __init__(self, expression, value=None, direction='max', *args, **kwargs):
        self._value = value
        self._direction = direction
        super(Objective, self).__init__(expression, *args, **kwargs)

    @property
    def value(self):
        return self._value

    def __str__(self):
        return {'max': 'Maximize', 'min': 'Minimize'}[self.direction] + '\n' + str(self.expression)
        # return ' '.join((self.direction, str(self.expression)))

    def _canonicalize(self, expression):
        """For example, changes x + y to 1.*x + 1.*y"""
        expression = super(Objective, self)._canonicalize(expression)
        expression *= 1.
        return expression

    @property
    def direction(self):
        """The direction of optimization. Either 'min' or 'max'."""
        return self._direction

    @direction.setter
    def direction(self, value):
        if value not in ['max', 'min']:
            raise ValueError("Provided optimization direction %s is neither 'min' or 'max'." % value)
        self._direction = value


class Configuration(object):
    """Optimization solver configuration."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def verbose(self):
        """Verbosity level.

        0: no output
        1: error and warning messages only
        2: normal output
        4: full output
        """
        raise NotImplementedError

    @verbose.setter
    def verbose(self, value):
        raise NotImplementedError

    @property
    def timeout(self):
        raise NotImplementedError

    @timeout.setter
    def timeout(self):
        raise NotImplementedError


class MathematicalProgrammingConfiguration(object):
    def __init__(self, *args, **kwargs):
        super(MathematicalProgrammingConfiguration, self).__init__(*args, **kwargs)

    @property
    def presolve(self):
        raise NotImplementedError

    @presolve.setter
    def presolve(self, value):
        raise NotImplementedError


class EvolutionaryOptimizationConfiguration(object):
    """docstring for HeuristicOptimization"""

    def __init__(self, *args, **kwargs):
        super(EvolutionaryOptimizationConfiguration, self).__init__(*args, **kwargs)


class Model(object):
    """Optimization problem.

    Attributes
    ----------
    objective: str
        The objective function.
    name: str, optional
        The name of the optimization problem.
    variables: Container, read-only
        Contains the variables of the optimization problem.
        The keys are the variable names and values are the actual variables.
    constraints: Container, read-only
         Contains the variables of the optimization problem.
         The keys are the constraint names and values are the actual constraints.
    status: str, read-only
        The status of the optimization problem.

    Examples
    --------


    """

    # TODO: configureation needs to be cloned too
    @classmethod
    def clone(cls, model):
        interface = sys.modules[cls.__module__]
        new_model = cls()
        for constraint in model.constraints:
            new_constraint = interface.Constraint.clone(constraint, model=new_model)
            new_model._add_constraint(new_constraint)
        if model.objective is not None:
            new_model.objective = interface.Objective.clone(model.objective, model=new_model)
        return new_model

    def __init__(self, name=None, objective=None, variables=None, constraints=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._objective = objective
        self._variables = Container()
        self._constraints = Container()
        self._variables_to_constraints_mapping = dict()
        self._status = None
        self.name = name
        if variables is not None:
            self.add(variables)
        if constraints is not None:
            self.add(constraints)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        try:
            for atom in value.expression.atoms(sympy.Symbol):
                if isinstance(atom, Variable) and (atom.problem is None or atom.problem != self):
                    self._add_variable(atom)
        except AttributeError as e:
            if isinstance(value.expression, six.types.FunctionType) or isinstance(value.expression, float):
                pass
            else:
                raise AttributeError(e)
        self._objective = value
        self._objective.problem = self

    @property
    def variables(self):
        return self._variables

    @property
    def constraints(self):
        return self._constraints

    @property
    def status(self):
        return self._status

    def __str__(self):
        return '\n'.join((
            str(self.objective),
            "subject to",
            '\n'.join([str(constr) for constr in self.constraints]),
            'Bounds',
            '\n'.join([str(var) for var in self.variables])
        ))

    @property
    def interface(self):
        return sys.modules[self.__module__]

    def add(self, stuff):
        """Add variables and constraints.

        Parameters
        ----------
        stuff : iterable, Variable, Constraint
            Either an iterable containing variables and constraints or a single variable or constraint.

        Returns
        -------
        None


        """
        if isinstance(stuff, collections.Iterable):
            for elem in stuff:
                self.add(elem)
        elif isinstance(stuff, Variable):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot add Variable %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            self._add_variable(stuff)
        elif isinstance(stuff, Constraint):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot add Constraint %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            self._add_constraint(stuff)
        elif isinstance(stuff, Objective):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot set Objective %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            self.objective = stuff
        else:
            raise TypeError("Cannot add %s. It is neither a Variable, Constraint, or Objective." % stuff)

    def remove(self, stuff):
        """Remove variables and constraints.

        Parameters
        ----------
        stuff : iterable, str, Variable, Constraint
            Either an iterable containing variables and constraints to be removed from the model or a single variable or contstraint (or their names).

        Returns
        -------
        None
        """
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
        elif isinstance(stuff, Variable):
            self._remove_variable(stuff)
        elif isinstance(stuff, Constraint):
            self._remove_constraint(stuff)
        elif isinstance(stuff, collections.Iterable):
            element_types = len(set((elem.__class__ for elem in stuff)))
            if len(element_types) == 1:
                element_type = element_types.pop()
                if element_type == Variable:
                    self._remove_variables(stuff)
                elif element_type == Constraint:
                    self._remove_constraints(stuff)
                else:
                    raise TypeError("Cannot remove %s. It neither a variable or constraint." % stuff)
            else:
                for elem in stuff:
                    self.remove(elem)
        elif isinstance(stuff, Objective):
            raise TypeError(
                "Cannot remove objective %s. Use model.objective = Objective(...) to change the current objective." % stuff)
        else:
            raise TypeError(
                "Cannot remove %s. It neither a variable or constraint." % stuff)

    def optimize(self):
        """Solve the optimization problem.

        Returns
        -------
        status: str
            Solution status.
        """
        raise NotImplementedError(
            "You're using the high level interface to optlang. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    def _add_variable(self, variable):
        self.variables.append(variable)
        self._variables_to_constraints_mapping[variable.name] = set([])
        variable.problem = self

        return variable

    def _remove_variables(self, variables):

        for variable in variables:
            try:
                var = self.variables[variable.name]
            except KeyError:
                raise LookupError("Variable %s not in solver" % var)

        constraint_ids = set()
        for variable in variables:
            constraint_ids.update(self._variables_to_constraints_mapping[variable.name])
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self.variables[variable.name]

        replacements = dict([(variable, 0) for variable in variables])
        for constraint_id in constraint_ids:
            constraint = self.constraints[constraint_id]
            constraint._expression = constraint.expression.xreplace(replacements)

        self.objective._expression = self.objective.expression.xreplace(replacements)

    def _remove_variable(self, variable):
        self._remove_variables([variable])

    def _add_constraint(self, constraint, sloppy=False):
        constraint_id = constraint.name
        if sloppy is False:
            for var in constraint.variables:
                if var.problem is not self:
                    self._add_variable(var)
                try:
                    self._variables_to_constraints_mapping[var.name].add(constraint_id)
                except KeyError:
                    self._variables_to_constraints_mapping[var.name] = set([constraint_id])
        self.constraints.append(constraint)
        constraint._problem = self

    def _remove_constraints(self, constraints):
        keys = [constraint.name for constraint in constraints]
        if len(constraints) > 350:  # Need to figure out a good threshold here
            self._constraints = self.constraints.fromkeys(set(self.constraints.keys()).difference(set(keys)))
        else:
            for constraint in constraints:
                try:
                    del self.constraints[constraint.name]
                except KeyError:
                    raise LookupError("Constraint %s not in solver" % constraint)
                else:
                    constraint.problem = None

    def _remove_constraint(self, constraint):
        self._remove_constraints([constraint])

    def _set_linear_objective_term(self, variable, coefficient):
        # TODO: the is extremely slow for objectives with many terms
        if variable in self.objective.expression.atoms(sympy.Symbol):
            a = sympy.Wild('a', exclude=[variable])
            (new_expression, map) = self.objective.expression.replace(lambda expr: expr.match(a*variable), lambda expr: coefficient*variable, simultaneous=False, map=True)
            self.objective.expression = new_expression
        else:
            self.objective.expression = sympy.Add._from_args((self.objective.expression, sympy.Mul._from_args((sympy.RealNumber(coefficient), variable))))

if __name__ == '__main__':
    # Example workflow

    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)
    c1 = Constraint(x1 + x2 + x3, ub=100)
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])

    try:
        sol = model.optimize()
    except NotImplementedError as e:
        print(e)

    print(model)
    print(model.variables)

    # model.remove(x1)

    import optlang

    model.interface = optlang.glpk_interface
