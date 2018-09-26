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
from weakref import WeakKeyDictionary

from six import iteritems
from sympy import Le, Dummy, oo
from sympy.solvers.inequalities import reduce_inequalities

from optlang.symbols import UniqueSymbol

__all__ = ("SymbolicParameter",)

LOGGER = logging.getLogger(__name__)


class SymbolicParameter(UniqueSymbol):
    """
    Instantiate a symbolic parameter to be used in bounds and constraints.

    A `SymbolicParameter` instance can be subscribed to and it will publish
    updates to the observing expressions. It can also be unsubscribed from. It
    thus follows the common Observer design pattern [1]_.

    Attributes
    ----------
    value : numeric
        The current numeric value of the parameter. Changing the value will
        update all expression this parameter is part of.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Observer_pattern

    """

    __slots__ = (
        "_observers",
        "_value"
    )

    def __init__(self, name, value=0, **kwargs):
        super(SymbolicParameter, self).__init__(name=name, **kwargs)
        self._observers = WeakKeyDictionary()
        self._value = value

    def __str__(self):
        """
        Return a string representation of the parameter.

        Examples
        --------
        >>> SymbolicParameter('mu')
        'mu'
        """
        return self.name

    def __repr__(self):
        """
        Return a full representation of the parameter.

        Examples
        --------
        >>> SymbolicParameter('mu', value=10)
        "<SymbolicParameter 'mu = 10'>"
        """
        return "<{} '{} = {}'>".format(
            type(self).__name__, self.name, self.value)

    @property
    def value(self):
        """Return the associated numeric value."""
        return self._value

    @value.setter
    def value(self, other):
        """Set a new value and update all observers."""
        self._value = other
        for observer, attributes in iteritems(self._observers):
            for attr in attributes:
                try:
                    observer.update(attr)
                except ReferenceError:
                    LOGGER.warning(
                        "The referenced observer no longer exists. This "
                        "suggests an inconsistent state.")
                    continue

    def attach(self, observer, attr):
        """Let an object and its expression observe this parameter."""
        self._observers.setdefault(observer, set()).add(attr)

    def detach(self, observer, attr):
        """Unregister an object from this instance."""
        self._observers[observer].remove(attr)

    def compute_bounds(self, substitute=False):
        """
        Compute the valid parameter bounds considering all inequalities.

        Parameters
        ----------
        substitute : bool, optional
            Whether or not to substitute the numeric values of other symbolic
            parameters.

        Returns
        -------

        """
        inequalities = list()
        bounds = frozenset(["lb", "ub", "bounds"])
        for observer, attributes in iteritems(self._observers):
            if len(bounds & attributes) == 0:
                # The symbolic parameter is not part of the bounds.
                continue
            lower = -oo if observer.lb is None else observer.lb
            upper = oo if observer.ub is None else observer.ub
            try:
                inequalities.append(Le(lower, upper))
            except ReferenceError:
                continue
        if len(inequalities) == 0:
            raise ValueError(
                "The symbolic parameter '{}' is not set on any bounds."
                "".format(self.name))
        # Reducing inequalities is currently not possible in symengine. Thus
        # we replace them with `sympy.Symbol`s here.
        var2sym = dict()
        for i, expr in enumerate(inequalities):
            if substitute:
                inequalities[i] = expr.subs({
                    sym: sym.value for sym in expr.atoms(SymbolicParameter)
                    if sym is not self})
            for var in expr.atoms(UniqueSymbol):
                if var not in var2sym:
                    var2sym[var] = Dummy(var.name)
            inequalities[i] = expr.subs({
                var: var2sym[var] for var in expr.atoms(UniqueSymbol)})
        return reduce_inequalities(inequalities, var2sym[self])
