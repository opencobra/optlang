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

from weakref import ref

from six import PY2

from optlang.symbols import UniqueSymbol
from optlang.util import is_numeric

__all__ = ("Variable",)


# noinspection PyShadowingBuiltins
class Variable(UniqueSymbol):
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

    # Might want to consider the use of slots in future for memory efficiency.
    __slots__ = ("_name", "_type", "_lb", "_ub", "_observer", "_observable")

    _TYPES = frozenset(["continuous", "integer", "binary"])

    def __init__(self, name, lb=None, ub=None, type="continuous", **kwargs):
        # Ensure that name is str and not binary or unicode.
        # Some solvers only support the `str` type in Python 2.
        if PY2:
            name = str(name)
        if len(name) == 0:
            raise ValueError("The variable's name must not be empty.")
        if any(char.isspace() for char in name):
            raise ValueError(
                "The variable's name cannot contain whitespace characters.")
        super(Variable, self).__init__(name=name, **kwargs)
        self._name = None
        self._type = None
        self._lb = None
        self._ub = None
        self._observer = None
        self._observable = None
        self.name = name
        self.type = type
        self.bounds = lb, ub

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

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __reduce__(self):
        return (type(self), (
            self.name, self.lb, self.ub, self.type))

    @staticmethod
    def _check_binary(value):
        if not (value is None or value == 0 or value == 1):
            raise ValueError(
                "Binary variable's bounds must be 0 or 1, not {}."
                "".format(value))

    @staticmethod
    def _check_bounds(lb, ub):
        if lb is None or ub is None:
            return
        if lb > ub:
            raise ValueError(
                "Lower bound must be smaller or equal to upper bound "
                "({} <= {}).".format(lb, ub))

    @classmethod
    def clone(cls, variable, **kwargs):
        """
        Make a copy of a variable.

        Example
        ----------
        >>> old_var = Variable("x")
        >>> var_copy = Variable.clone(old_var)
        """
        return cls(name=variable.name, lb=variable.lb, ub=variable.ub,
                   type=variable.type, **kwargs)

    @property
    def name(self):
        """Name of variable."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        observer = getattr(self, "_observer", None)
        try:
            observer().update_variable_name(self, value)
        except (AttributeError, TypeError):
            # Observer is not set or no longer exists.
            pass

    @property
    def type(self):
        """The variable's type (either 'continuous', 'integer', or 'binary')."""
        return self._type

    @type.setter
    def type(self, value):
        if value not in self._TYPES:
            raise ValueError(
                "'{}' is not a recognized variable type. Choose between {}."
                "".format(value, ", ".join(self._TYPES)))
        self._type = value
        try:
            self._observer().update_variable_type(self, value)
        except (AttributeError, TypeError):
            # Observer is not set or no longer exists.
            pass

    @property
    def lb(self):
        """Lower bound of variable."""
        return self._lb

    @lb.setter
    def lb(self, value):
        # TODO: Require method for numeric values for these checks.
        if self._type == "binary":
            self._check_binary(value)
        self._check_bounds(value, self._ub)
        self._lb = value
        try:
            self._observer().update_variable_lb(self, value)
        except (AttributeError, TypeError):
            # Observer is not set or no longer exists.
            pass

    @property
    def ub(self):
        """Upper bound of variable."""
        return self._ub

    @ub.setter
    def ub(self, value):
        # TODO: Require method for numeric values for these checks.
        if self._type == "binary":
            self._check_binary(value)
        self._check_bounds(self._lb, value)
        self._ub = value
        try:
            self._observer().update_variable_ub(self, value)
        except (AttributeError, TypeError):
            # Observer is not set or no longer exists.
            pass

    @property
    def bounds(self):
        """The variable's lower and upper bound."""
        return self._lb, self._ub

    @bounds.setter
    def bounds(self, pair):
        lb, ub = pair
        # TODO: Require method for numeric values for these checks.
        if self._type == "binary":
            self._check_binary(lb)
            self._check_binary(ub)
        self._check_bounds(lb, ub)
        self._lb = lb
        self._ub = ub
        try:
            self._observer().update_variable_bounds(self, self._lb, self._ub)
        except (AttributeError, TypeError):
            # Observer is not set or no longer exists.
            pass

    @property
    def primal(self):
        """The primal of variable (None if no solution exists)."""
        try:
            return self._observable().get_variable_primal(self)
        except (AttributeError, TypeError):
            # Observable is not set or no longer exists.
            return None

    @property
    def dual(self):
        """The dual of variable (None if no solution exists)."""
        try:
            return self._observable().get_variable_dual(self)
        except (AttributeError, TypeError):
            # Observable is not set or no longer exists.
            return None

    def set_observer(self, observer):
        self._observer = ref(observer)

    def set_observable(self, observable):
        self._observable = ref(observable)

    def to_dict(self):
        """
        Return an object representation of the variable.

        Example
        --------
        >>> import json
        >>> var = Variable("x")
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
        Construct a variable from the provided object.

        Parameters
        ----------
        obj : dict
            A variable representation as returned by ``Variable.to_dict``.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     var = Variable.from_dict(json.load(infile))
        """
        return cls(**obj)
