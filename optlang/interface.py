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
import collections
import inspect
import logging
import random
import sys
import uuid

import six

from optlang.exceptions import IndicatorConstraintsNotSupported

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

    @staticmethod
    def __test_valid_lower_bound(type, value, name):
        if value is not None:
            if type == 'integer' and value % 1 != 0.:
                raise ValueError(
                    'The provided lower bound %g cannot be assigned to integer variable %s (%g mod 1 != 0).' % (
                        value, name, value))
        if type == 'binary' and (value is None or value != 0):
            raise ValueError(
                'The provided lower bound %s cannot be assigned to binary variable %s.' % (value, name))

    @staticmethod
    def __test_valid_upper_bound(type, value, name):
        if value is not None:
            if type == 'integer' and value % 1 != 0.:
                raise ValueError(
                    'The provided upper bound %s cannot be assigned to integer variable %s (%s mod 1 != 0).' % (
                        value, name, value))
        if type == 'binary' and (value is None or value != 1):
            raise ValueError(
                'The provided upper bound %s cannot be assigned to binary variable %s.' % (value, name))

    @classmethod
    def clone(cls, variable, **kwargs):
        """Clone another variable (for example from another solver interface)."""
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
        return sympy.Symbol.__xnew__(cls, name, uuid=str(int(round(1e16 * random.random()))),
                                     **assumptions)  # uuid.uuid1()

    def __init__(self, name, lb=None, ub=None, type="continuous", problem=None, *args, **kwargs):

        # Ensure that name is str and not binary of unicode - some solvers only support string type in Python 2.
        if six.PY2:
            name = str(name)

        for char in name:
            if char.isspace():
                raise ValueError(
                    'Variable names cannot contain whitespace characters. "%s" contains whitespace character "%s".' % (
                        name, char))
        self._name = name
        sympy.Symbol.__init__(name, *args, **kwargs)
        self._lb = lb
        self._ub = ub
        if self._lb is None and type == 'binary':
            self._lb = 0.
        if self._ub is None and type == 'binary':
            self._ub = 1.
        self.__test_valid_lower_bound(type, self._lb, name)
        self.__test_valid_upper_bound(type, self._ub, name)
        self._type = type
        self.problem = problem

    @property
    def name(self):
        """Name of variable."""
        return self._name

    @name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        self._name = value
        if getattr(self, 'problem', None) is not None:
            self.problem.variables.update_key(old_name)

    @property
    def lb(self):
        """Lower bound of variable."""
        return self._lb

    @lb.setter
    def lb(self, value):
        if hasattr(self, 'ub') and self.ub is not None and value is not None and value > self.ub:
            raise ValueError(
                'The provided lower bound %g is larger than the upper bound %g of variable %s.' % (
                    value, self.ub, self))
        self.__test_valid_lower_bound(self.type, value, self.name)
        self._lb = value
        if self.problem is not None:
            self.problem._pending_modifications.var_lb.append((self, value))

    @property
    def ub(self):
        """Upper bound of variable."""
        return self._ub

    @ub.setter
    def ub(self, value):
        if hasattr(self, 'lb') and self.lb is not None and value is not None and value < self.lb:
            raise ValueError(
                'The provided upper bound %g is smaller than the lower bound %g of variable %s.' % (
                    value, self.lb, self))
        self.__test_valid_upper_bound(self.type, value, self.name)
        self._ub = value
        if self.problem is not None:
            self.problem._pending_modifications.var_ub.append((self, value))

    @property
    def type(self):
        """Variable type ('either continuous, integer, or binary'.)"""
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

    @property
    def primal(self):
        """The primal of variable (None if no solution exists)."""
        return None

    @property
    def dual(self):
        """The dual of variable (None if no solution exists)."""
        return None

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

    def _round_primal_to_bounds(self, primal, tolerance=1e-5):
        """Rounds primal value to lie within variables bounds.

        Raises if exceeding threshold.

        Parameters
        ----------
        primal : float
            The primal value to round.
        tolerance : float (optional)
            The tolerance threshold (default: 1e-5).
        """
        if (self.lb is None or primal >= self.lb) and (self.ub is None or primal <= self.ub):
            return primal
        else:
            if (primal <= self.lb) and ((self.lb - primal) <= tolerance):
                return self.lb
            elif (primal >= self.ub) and ((self.ub - primal) >= -tolerance):
                return self.ub
            else:
                raise AssertionError(
                    'The primal value %s returned by the solver is out of bounds for variable %s (lb=%s, ub=%s)' % (
                        primal, self.name, self.lb, self.ub))


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
        # Ensure that name is str and not binary of unicode - some solvers only support string type in Python 2.
        if six.PY2 and name is not None:
            name = str(name)

        super(OptimizationExpression, self).__init__(*args, **kwargs)
        self._problem = problem
        if sloppy:
            self._expression = expression
        else:
            self._expression = self._canonicalize(expression)
        if name is None:
            self._name = str(uuid.uuid1())
        else:
            self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

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
            # expression = expression.expand() This would be a good way to canonicalize, but is quite slow
            return expression

    @property
    def is_Linear(self):
        """Returns True if constraint is linear (read-only)."""
        coeff_dict = self.expression.as_coefficients_dict()
        if all((len(key.free_symbols) < 2 and (key.is_Add or key.is_Mul or key.is_Atom) for key in coeff_dict.keys())):
            return True
        else:
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
        """Returns True if constraint is quadratic (read-only)."""
        if self.expression.is_Atom:
            return False
        if all((len(key.free_symbols) < 2 and (key.is_Add or key.is_Mul or key.is_Atom)
                for key in self.expression.as_coefficients_dict().keys())):
            return False
        try:
            if self.expression.is_Add:
                terms = self.expression.args
                is_quad = False
                for term in terms:
                    if len(term.free_symbols) > 2:
                        return False
                    if term.is_Pow:
                        if not term.args[1].is_Number or term.args[1] > 2:
                            return False
                        else:
                            is_quad = True
                    elif term.is_Mul:
                        if len(term.free_symbols) == 2:
                            is_quad = True
                        if term.args[1].is_Pow:
                            if not term.args[1].args[1].is_Number or term.args[1].args[1] > 2:
                                return False
                            else:
                                is_quad = True
                return is_quad
            else:
                return self.expression.as_poly(*self.variables).is_quadratic
        except sympy.PolynomialError:
            return False

    def __iadd__(self, other):
        self._expression += other
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
    indicator_variable: Variable
        The indicator variable (needs to be binary).
    active_when: 0 or 1 (default 0)
        When the constraint should
    problem: Model or None, optional
        A reference to the optimization model the variable belongs to.
    """

    _INDICATOR_CONSTRAINT_SUPPORT = True

    @classmethod
    def __check_valid_indicator_variable(cls, variable):
        if variable is not None and not cls._INDICATOR_CONSTRAINT_SUPPORT:
            raise IndicatorConstraintsNotSupported(
                'Solver interface %s does not support indicator constraints' % cls.__module__)
        if variable is not None and variable.type != 'binary':
            raise ValueError('Provided indicator variable %s is not binary.' % variable)

    @staticmethod
    def __check_valid_active_when(active_when):
        if active_when != 0 and active_when != 1:
            raise ValueError('Provided active_when argument %s needs to be either 1 or 0' % active_when)

    @classmethod
    def clone(cls, constraint, model=None, **kwargs):
        """Clone another constraint (for example from another solver interface)."""
        return cls(cls._substitute_variables(constraint, model=model), lb=constraint.lb, ub=constraint.ub,
                   indicator_variable=constraint.indicator_variable, active_when=constraint.active_when,
                   name=constraint.name, sloppy=True, **kwargs)

    def __init__(self, expression, lb=None, ub=None, indicator_variable=None, active_when=1, *args, **kwargs):
        self._lb = lb
        self._ub = ub
        super(Constraint, self).__init__(expression, *args, **kwargs)
        self.__check_valid_indicator_variable(indicator_variable)
        self.__check_valid_active_when(active_when)
        self._indicator_variable = indicator_variable
        self._active_when = active_when

    @property
    def lb(self):
        """Lower bound of constraint."""
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value

    @property
    def ub(self):
        """Upper bound of constraint."""
        return self._ub

    @ub.setter
    def ub(self, value):
        self._ub = value

    @property
    def indicator_variable(self):
        """The indicator variable of constraint (if available)."""
        return self._indicator_variable

    @indicator_variable.setter
    def indicator_variable(self, value):
        self.__check_valid_indicator_variable(value)
        self._indicator_variable = value

    @property
    def active_when(self):
        """Activity relation of constraint to indicator variable (if supported)."""
        return self._active_when

    @indicator_variable.setter
    def indicator_variable(self, value):
        self.__check_valid_active_when(value)
        self._active_when = value

    def __str__(self):
        if self.lb is not None:
            lhs = str(self.lb) + ' <= '
        else:
            lhs = ''
        if self.ub is not None:
            rhs = ' <= ' + str(self.ub)
        else:
            rhs = ''
        if self.indicator_variable is not None:
            lhs = self.indicator_variable.name + ' = ' + str(self.active_when) + ' -> ' + lhs
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
        else:
            expression = expression - coeff
            if self.lb is not None:
                self.lb = self.lb - coeff
            if self.ub is not None:
                self.ub = self.ub - coeff
        return expression

    @property
    def primal(self):
        """Primal of constraint (None if no solution exists)."""
        return None

    @property
    def dual(self):
        """Dual of constraint (None if no solution exists)."""
        return None

    def _round_primal_to_bounds(self, primal, tolerance=1e-5):
        if (self.lb is None or primal >= self.lb) and (self.ub is None or primal <= self.ub):
            return primal
        else:
            if (primal <= self.lb) and ((self.lb - primal) <= tolerance):
                return self.lb
            elif (primal >= self.ub) and ((self.ub - primal) >= -tolerance):
                return self.ub
            else:
                raise AssertionError(
                    'The primal value %s returned by the solver is out of bounds for variable %s (lb=%s, ub=%s)' % (
                        primal, self.name, self.lb, self.ub))


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
        """Clone another objective (for example from another solver interface)."""
        return cls(cls._substitute_variables(objective, model=model), name=objective.name,
                   direction=objective.direction, sloppy=True, **kwargs)

    def __init__(self, expression, value=None, direction='max', *args, **kwargs):
        self._value = value
        self._direction = direction
        super(Objective, self).__init__(expression, *args, **kwargs)

    @property
    def value(self):
        """The objective value."""
        return self._value

    def __str__(self):
        return {'max': 'Maximize', 'min': 'Minimize'}[self.direction] + '\n' + str(self.expression)
        # return ' '.join((self.direction, str(self.expression)))

    def __eq__(self, other):
        """Tests *mathematical* equality for two Objectives. Solver specific type does NOT have to match.
        Expression and direction must be the same.
        Name does not have to match"""
        if isinstance(other, Objective):
            return self.expression == other.expression and self.direction == other.direction
        else:
            return False
            #

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

    @classmethod
    def clone(cls, config, problem=None, **kwargs):
        properties = (k for k, v in inspect.getmembers(cls, predicate=inspect.isdatadescriptor) if
                      not k.startswith('__'))
        parameters = {property: getattr(config, property) for property in properties if hasattr(config, property)}
        return cls(problem=problem, **parameters)

    def __init__(self, problem=None, *args, **kwargs):
        self.problem = problem

    @property
    def verbosity(self):
        """Verbosity level.

        0: no output
        1: error and warning messages only
        2: normal output
        4: full output
        """
        raise NotImplementedError

    @verbosity.setter
    def verbosity(self, value):
        raise NotImplementedError

    @property
    def timeout(self):
        """Timeout parameter (seconds)."""
        raise NotImplementedError

    @timeout.setter
    def timeout(self):
        raise NotImplementedError


class MathematicalProgrammingConfiguration(Configuration):
    def __init__(self, *args, **kwargs):
        super(MathematicalProgrammingConfiguration, self).__init__(*args, **kwargs)

    @property
    def presolve(self):
        """If the presolver should be used (if available)."""
        raise NotImplementedError

    @presolve.setter
    def presolve(self, value):
        raise NotImplementedError


class EvolutionaryOptimizationConfiguration(Configuration):
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

    @classmethod
    def clone(cls, model):
        """Clone another model (for example from another solver interface)."""
        interface = sys.modules[cls.__module__]
        new_model = cls()
        for variable in model.variables:
            new_variable = interface.Variable.clone(variable)
            new_model._add_variable(new_variable)
        for constraint in model.constraints:
            new_constraint = interface.Constraint.clone(constraint, model=new_model)
            new_model._add_constraint(new_constraint)
        if model.objective is not None:
            new_model.objective = interface.Objective.clone(model.objective, model=new_model)
        new_model.configuration = interface.Configuration.clone(model.configuration, problem=new_model)
        return new_model

    def __init__(self, name=None, objective=None, variables=None, constraints=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._objective = objective
        self._variables = Container()
        self._constraints = Container()
        self._variables_to_constraints_mapping = dict()
        self._status = None

        class Modifications():

            def __init__(self):
                self.add_var = []
                self.add_constr = []
                self.add_constr_sloppy = []
                self.rm_var = []
                self.rm_constr = []
                self.var_lb = []
                self.var_ub = []
                self.toggle = 'add'

            def __str__(self):
                return str(self.__dict__)

        self._pending_modifications = Modifications()
        self.name = name
        if variables is not None:
            self.add(variables)
        if constraints is not None:
            self.add(constraints)

    @property
    def interface(self):
        """Provides access to the solver interface the model belongs to

        For example optlang.glpk_interface
        """
        return sys.modules[self.__module__]

    @property
    def objective(self):
        """The model's objective function."""
        return self._objective

    @objective.setter
    def objective(self, value):
        self.update()
        try:
            for atom in value.expression.atoms(sympy.Symbol):
                if isinstance(atom, Variable) and (atom.problem is None or atom.problem != self):
                    print(atom, atom.problem)
                    self._pending_modifications.add_var.append(atom)
            self.update()
        except AttributeError as e:
            if isinstance(value.expression, six.types.FunctionType) or isinstance(value.expression, float):
                pass
            else:
                raise AttributeError(e)
        if self._objective is not None:
            self._objective.problem = None
        self._objective = value
        self._objective.problem = self

    @property
    def variables(self):
        """The model variables."""
        self.update()
        return self._variables

    @property
    def constraints(self):
        """The model constraints."""
        self.update()
        return self._constraints

    @property
    def status(self):
        """The solver status of the model."""
        return self._status

    @property
    def primal_values(self):
        """The primal values of model/all variables.

        Returns
        -------
        collections.OrderedDict
        """
        # Fallback, if nothing faster is available
        return collections.OrderedDict([(variable.name, variable.primal) for variable in self.variables])

    @property
    def reduced_costs(self):
        """The reduced costs/dual values of all variables.

        Returns
        -------
        collections.OrderedDict
        """
        # Fallback, if nothing faster is available
        return collections.OrderedDict([(variable.name, variable.dual) for variable in self.variables])

    @property
    def dual_values(self):
        """The dual values of the problem (primal values of all constraints).

        Returns
        -------
        collections.OrderedDict
        """
        # Fallback, if nothing faster is available
        return collections.OrderedDict([(constraint.name, constraint.primal) for constraint in self.constraint])

    @property
    def shadow_prices(self):
        """The shadow prices of model (dual values of all variables).

        Returns
        -------
        collections.OrderedDict
        """
        # Fallback, if nothing faster is available
        return collections.OrderedDict([(constraint.name, constraint.dual) for constraint in self.constraint])

    def __str__(self):
        self.update()
        return '\n'.join((
            str(self.objective),
            "subject to",
            '\n'.join([str(constr) for constr in self.constraints]),
            'Bounds',
            '\n'.join([str(var) for var in self.variables])
        ))

    def add(self, stuff, sloppy=False):
        """Add variables and constraints.

        Parameters
        ----------
        stuff : iterable, Variable, Constraint
            Either an iterable containing variables and constraints or a single variable or constraint.

        sloppy : bool
            Check constraints for variables that are not part of the model yet.

        Returns
        -------
        None
        """
        if self._pending_modifications.toggle == 'remove':
            self.update()
            self._pending_modifications.toggle = 'add'
        if isinstance(stuff, collections.Iterable):
            for elem in stuff:
                self.add(elem, sloppy=sloppy)
        elif isinstance(stuff, Variable):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot add Variable %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            self._pending_modifications.add_var.append(stuff)
        elif isinstance(stuff, Constraint):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot add Constraint %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            if sloppy is True:
                self._pending_modifications.add_constr_sloppy.append(stuff)
            else:
                self._pending_modifications.add_constr.append(stuff)
        else:
            raise TypeError("Cannot add %s. It is neither a Variable, or Constraint." % stuff)

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
        if self._pending_modifications.toggle == 'add':
            self.update()
            self._pending_modifications.toggle = 'remove'
        if isinstance(stuff, str):
            try:
                variable = self.variables[stuff]
                self._pending_modifications.rm_var.append(variable)
            except KeyError:
                try:
                    constraint = self.constraints[stuff]
                    self._pending_modifications.rm_constr.append(constraint)
                except KeyError:
                    raise LookupError(
                        "%s is neither a variable nor a constraint in the current solver instance." % stuff)
        elif isinstance(stuff, Variable):
            self._pending_modifications.rm_var.append(stuff)
        elif isinstance(stuff, Constraint):
            self._pending_modifications.rm_constr.append(stuff)
        elif isinstance(stuff, collections.Iterable):
            for elem in stuff:
                self.remove(elem)
        elif isinstance(stuff, Objective):
            raise TypeError(
                "Cannot remove objective %s. Use model.objective = Objective(...) to change the current objective." % stuff)
        else:
            raise TypeError(
                "Cannot remove %s. It neither a variable or constraint." % stuff)

    def update(self):
        """Process all pending model modifications."""
        # print(self._pending_modifications)
        add_var = self._pending_modifications.add_var
        if len(add_var) > 0:
            self._add_variables(add_var)
            self._pending_modifications.add_var = []

        add_constr = self._pending_modifications.add_constr
        if len(add_constr) > 0:
            self._add_constraints(add_constr)
            self._pending_modifications.add_constr = []

        add_constr_sloppy = self._pending_modifications.add_constr_sloppy
        if len(add_constr_sloppy) > 0:
            self._add_constraints(add_constr_sloppy, sloppy=True)
            self._pending_modifications.add_constr_sloppy = []

        var_lb = self._pending_modifications.var_lb
        var_ub = self._pending_modifications.var_ub
        if len(var_lb) > 0 or len(var_ub) > 0:
            self._set_variable_bounds_on_problem(var_lb, var_ub)
            self._pending_modifications.var_lb = []
            self._pending_modifications.var_ub = []

        rm_var = self._pending_modifications.rm_var
        if len(rm_var) > 0:
            self._remove_variables(rm_var)
            self._pending_modifications.rm_var = []

        rm_constr = self._pending_modifications.rm_constr
        if len(rm_constr) > 0:
            self._remove_constraints(rm_constr)
            self._pending_modifications.rm_constr = []

    def optimize(self):
        """Solve the optimization problem.

        Returns
        -------
        status: str
            Solution status.
        """
        self.update()
        status = self._optimize()
        if status != OPTIMAL and self.configuration.presolve == "auto":
            self.configuration.presolve = True
            status = self._optimize()
            self.configuration.presolve = "auto"
        return status

    def _optimize(self):
        raise NotImplementedError(
            "You're using the high level interface to optlang. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        """"""
        pass

    def _add_variable(self, variable):
        self._add_variables([variable])

    def _add_variables(self, variables):
        for variable in variables:
            self._variables.append(variable)
            self._variables_to_constraints_mapping[variable.name] = set([])
            variable.problem = self

    def _remove_variables(self, variables):
        for variable in variables:
            try:
                var = self._variables[variable.name]
            except KeyError:
                raise LookupError("Variable %s not in solver" % var)

        constraint_ids = set()
        for variable in variables:
            constraint_ids.update(self._variables_to_constraints_mapping[variable.name])
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]

        replacements = dict([(variable, 0) for variable in variables])
        for constraint_id in constraint_ids:
            constraint = self._constraints[constraint_id]
            constraint._expression = constraint.expression.xreplace(replacements)
        if self.objective is not None:
            self.objective._expression = self.objective.expression.xreplace(replacements)

    def _remove_variable(self, variable):
        self._remove_variables([variable])

    def _add_constraint(self, constraint, sloppy=False):
        self._add_constraints([constraint], sloppy=sloppy)

    def _add_constraints(self, constraints, sloppy=False):
        for constraint in constraints:
            constraint_id = constraint.name
            if sloppy is False:
                variables = constraint.variables
                if constraint.indicator_variable is not None:
                    variables.add(constraint.indicator_variable)
                missing_vars = [var for var in variables if var.problem is not self]
                if len(missing_vars) > 0:
                    self._add_variables(missing_vars)
                for var in variables:
                    try:
                        self._variables_to_constraints_mapping[var.name].add(constraint_id)
                    except KeyError:
                        self._variables_to_constraints_mapping[var.name] = set([constraint_id])
            self._constraints.append(constraint)
            constraint._problem = self

    def _remove_constraints(self, constraints):
        for constraint in constraints:  # TODO: remove this hack that fixes problems with lazy solver expressions.
            constraint.expression
        keys = [constraint.name for constraint in constraints]
        if len(constraints) > 350:  # Need to figure out a good threshold here
            self._constraints = self.constraints.fromkeys(set(self.constraints.keys()).difference(set(keys)))
        else:
            for constraint in constraints:
                try:
                    del self._constraints[constraint.name]
                except KeyError:
                    raise LookupError("Constraint %s not in solver" % constraint)
                else:
                    constraint._problem = None

    def _remove_constraint(self, constraint):
        self._remove_constraints([constraint])

    def _set_linear_objective_term(self, variable, coefficient):
        if variable in self.objective.expression.atoms(sympy.Symbol):
            a = sympy.Wild('a', exclude=[variable])
            (new_expression, map) = self.objective.expression.replace(lambda expr: expr.match(a * variable),
                                                                      lambda expr: coefficient * variable,
                                                                      simultaneous=False, map=True)
            self.objective.expression = new_expression
        else:
            self.objective.expression = sympy.Add._from_args(
                (self.objective.expression, sympy.Mul._from_args((sympy.RealNumber(coefficient), variable))))


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
