# -*- coding: utf-8 -*-

# Copyright 2017 Novo Nordisk Foundation Center for Biosustainability,
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

from __future__ import absolute_import

from six import PY2

from optlang.symbolics import Symbol
from optlang.util import is_numeric

__all__ = ("Variable",)


# noinspection PyShadowingBuiltins
class Variable(Symbol):
    """Optimization variable.

    Variable objects are used to represents each variable of the optimization problem. When the optimization is
    performed, the combination of variable values that optimizes the objective function, while not violating any
    constraints will be identified. The type of a variable ('continuous', 'integer' or 'binary') can be set using
    the type keyword of the constructor or it can be changed after initialization by :code:`var.type = 'binary'`.

    The variable class subclasses the :code:`sympy.Symbol` class, which means that symbolic expressions of variables
    can be constructed by using regular python syntax, e.g. :code:`my_expression = 2 * var1 + 3 * var2 ** 2`.
    Expressions like this are used when constructing Constraint and Objective objects.
    Once a problem has been optimized, the primal and dual values of a variable can be accessed from the
    primal and dual attributes, respectively.

    Attributes
    ----------
    name: str
        The variable's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    type: {'continuous', 'integer', 'binary'}, optional
        The variable type, 'continuous' or 'integer' or 'binary'.
    subject: optlang.Model or None, optional
        A reference to the optimization model the variable belongs to.

    Examples
    --------
    >>> Variable('x', lb=-10, ub=10)
    '-10 <= x <= 10'
    """

    @staticmethod
    def __test_valid_lower_bound(type, value, name):
        if not (value is None or is_numeric(value)):
            raise TypeError("Variable bounds must be numeric or None.")
        if value is not None:
            if type == 'integer' and value % 1 != 0.:
                raise ValueError(
                    'The provided lower bound %g cannot be assigned to integer variable %s (%g mod 1 != 0).' % (
                        value, name, value))
        if type == 'binary' and (value is None or value not in (0, 1)):
            raise ValueError(
                'The provided lower bound %s cannot be assigned to binary variable %s.' % (value, name))

    @staticmethod
    def __test_valid_upper_bound(type, value, name):
        if not (value is None or is_numeric(value)):
            raise TypeError("Variable bounds must be numeric or None.")
        if value is not None:
            if type == 'integer' and value % 1 != 0.:
                raise ValueError(
                    'The provided upper bound %s cannot be assigned to integer variable %s (%s mod 1 != 0).' % (
                        value, name, value))
        if type == 'binary' and (value is None or value not in (0, 1)):
            raise ValueError(
                'The provided upper bound %s cannot be assigned to binary variable %s.' % (value, name))

    @classmethod
    def clone(cls, variable, **kwargs):
        """
        Make a copy of a variable.

        Example
        ----------
        >>> var_copy = Variable.clone(old_var)
        """
        return cls(variable.name, lb=variable.lb, ub=variable.ub,
                   type=variable.type, **kwargs)

    def __init__(self, name, lb=None, ub=None, type="continuous",
                 subject=None, **kwargs):
        self.subject = subject
        # Ensure that name is str and not binary or unicode.
        # Some solvers only support the `str` type in Python 2.
        if PY2:
            name = str(name)
        if len(name) == 0:
            raise ValueError('The variable name must not be empty.')
        if any(char.isspace() for char in name):
            raise ValueError(
                'Variable names cannot contain whitespace characters.')
        self._name = name
        super(Variable, self).__init__(self, name, **kwargs)
        self._lb = lb
        self._ub = ub
        if self._lb is None and type == 'binary':
            self._lb = 0.
        if self._ub is None and type == 'binary':
            self._ub = 1.
        self.__test_valid_lower_bound(type, self._lb, name)
        self.__test_valid_upper_bound(type, self._ub, name)
        self._type = None
        self.type = type

    @property
    def name(self):
        """Name of variable."""
        return self._name

    @name.setter
    def name(self, value):
        old_name = getattr(self, "name", "")
        self._name = value
        subject = getattr(self, "subject", None)
        if subject is not None and value != old_name:
            subject.update_variable(self, "name", value)

    @property
    def lb(self):
        """Lower bound of variable."""
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value
        if self.subject is not None:
            self.subject.update_variable(self, value)

    @property
    def ub(self):
        """Upper bound of variable."""
        return self._ub

    @ub.setter
    def ub(self, value):
        self._ub = value
        if self.subject is not None:
            self.subject.update_variable(self, "ub", value)

    @property
    def bounds(self):
        """The variable's lower and upper bound."""
        return self._lb, self._ub

    @bounds.setter
    def bounds(self, pair):
        self._lb = pair[0]
        self._ub = pair[1]
        if self.subject is not None:
            self.subject.update_variable(self, "lb", self._lb)
            self.subject.update_variable(self, "ub", self._ub)

    @property
    def type(self):
        """The variable's type (either 'continuous', 'integer', or 'binary')."""
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
                "'{}' is not a valid variable type. Choose between 'continuous,"
                " 'integer', or 'binary'.".format(value))

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
        lb_str = str(self.lb) if self.lb is not None else "-Inf"
        ub_str = str(self.ub) if self.ub is not None else "Inf"
        return '{} <= {} <= {}'.format(
            lb_str, super(Variable, self).__str__(), ub_str)

    def __repr__(self):
        """Does exactly the same as __str__ for now."""
        return self.__str__()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __reduce__(self):
        return (type(self), (
            self.name, self.lb, self.ub, self.type, self.subject))

    def to_dict(self):
        """
        Return an object representation of the Variable.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(var.to_dict(), outfile)
        """
        return {
            "name": self.name,
            "lb": self.lb,
            "ub": self.ub,
            "type": self.type
        }

    @classmethod
    def from_dict(cls, obj):
        """
        Construct a Variable from the provided object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     var = Variable.from_dict(json.load(infile))
        """
        return cls(name=obj["name"], lb=obj.get("lb"), ub=obj.get("ub"),
                   type=obj["type"])
