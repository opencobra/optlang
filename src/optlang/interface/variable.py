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

import logging
from enum import Enum, unique

from optlang.symbols import UniqueSymbol
from optlang.interface.mixins import (
    BoundsMixin, ValueMixin, NameMixin, SymbolicMixin)

__all__ = ("VariableType", "Variable")

LOGGER = logging.getLogger(__name__)


@unique
class VariableType(Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


# noinspection PyShadowingBuiltins
class Variable(NameMixin, BoundsMixin, ValueMixin, SymbolicMixin,
               UniqueSymbol):
    """
    Represent an optimization variable.

    `Variable` objects are used to represents each variable of the optimization
    problem. When the optimization is performed, the combination of variable
    values that optimizes the objective function, while not violating any
    constraints will be identified. The type of a variable ('continuous',
    'integer' or 'binary') can be set using the type keyword of the constructor
    or it can be changed after initialization by :code:`var.type = 'binary'`.

    The variable class inherits from the `UniqueSymbol` class, which means
    that symbolic expressions of variables can be constructed by using regular
    python syntax, e.g. :code:`my_expression = 2 * var1 + 3 * var2 ** 2`.
    Expressions like this are used when constructing `Constraint` and
    `Objective` objects.
    Once a problem has been optimized, the primal and dual values of a
    variable can be accessed from the primal and dual attributes, respectively.

    Attributes
    ----------
    name: str
        The variable's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    type: {'continuous', 'integer', 'binary'}, optional
        The variable type can be continuous, integer, or binary. The default
        is continuous.
    bounds: tuple
        The lower and upper bound as a pair `lb, ub`.

    Examples
    --------
    >>> Variable('x', lb=-10, ub=10)
    '-10 <= x <= 10'
    """

    __slots__ = (
        "_observer",
        "_solver",
        "_name",
        "_lb", "_numeric_lb", "_ub", "_numeric_ub",
        "_type",
        "__weakref__"
    )

    def __init__(self, name, lb=None, ub=None, type="continuous", **kwargs):
        """
        Initialize a variable.

        Parameters
        ----------
        name: str
            The variable's name.
        lb: float or None, optional
            The lower bound, if None then -inf.
        ub: float or None, optional
            The upper bound, if None then inf.
        type: {'continuous', 'integer', 'binary'}, optional
            The variable type can be continuous, integer, or binary. The default
            is continuous.

        """
        super(Variable, self).__init__(name=name, **kwargs)
        self._type = None
        self.name = name
        self.type = type
        self.bounds = lb, ub

    def __repr__(self):
        lb_str = str(self.lb) if self.lb is not None else "-Inf"
        ub_str = str(self.ub) if self.ub is not None else "Inf"
        return "<{} Variable '{} <= {} <= {}'>".format(
            self._type.value, lb_str, super(Variable, self).__str__(), ub_str)

    def __str__(self):
        """
        Return a string representation of the variable.

        Examples
        --------
        >>> Variable('x', lb=-10, ub=10)
        'x'
        """
        return super(Variable, self).__str__()

    def __reduce__(self):
        return (type(self), (
            self.name, self.lb, self.ub, self.type))

    @staticmethod
    def _check_binary(value):
        # Use equivalence here such that different numeric classes can be
        # compared, e.g., optlang.symbols.Integer.
        if not (value is None or value == 0 or value == 1):
            raise ValueError(
                "Binary variable's bounds must be None, 0, or 1 not: {}."
                "".format(value))

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
    def type(self):
        """Modify the variable's type."""
        return self._type

    @type.setter
    def type(self, value):
        self._type = VariableType(value)
        try:
            self._observer.update_type(self, self._type)
        except (AttributeError, ReferenceError):
            # Observer is not set or no longer exists.
            pass

    @BoundsMixin.lb.setter
    def lb(self, value):
        # Conversion to a numeric value is done at this place because if it
        # fails the attributes are not yet changed.
        numeric = self._evaluate(value)
        if self._type is VariableType.BINARY:
            self._check_binary(numeric)
        self._set_numeric_lb(numeric)
        self._disregard_symbols(self.lb, "lb")
        self._observe_symbols(value, "lb")
        self._lb = value

    @BoundsMixin.ub.setter
    def ub(self, value):
        # Conversion to a numeric value is done at this place because if it
        # fails the attributes are not yet changed.
        numeric = self._evaluate(value)
        if self._type is VariableType.BINARY:
            self._check_binary(numeric)
        self._disregard_symbols(self.ub, "ub")
        self._observe_symbols(value, "ub")
        self._ub = value
        self._set_numeric_ub(numeric)

    @BoundsMixin.bounds.setter
    def bounds(self, pair):
        lb, ub = pair
        # Conversion to a numeric value is done at this place because if it
        # fails the attributes are not yet changed.
        num_lb = self._evaluate(lb)
        num_ub = self._evaluate(ub)
        if self._type is VariableType.BINARY:
            self._check_binary(num_lb)
            self._check_binary(num_ub)
        self._set_numeric_bounds(num_lb, num_ub)
        self._disregard_symbols(self.lb, "bounds")
        self._disregard_symbols(self.ub, "bounds")
        self._observe_symbols(lb, "bounds")
        self._observe_symbols(ub, "bounds")
        self._lb = lb
        self._ub = ub

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
