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

from optlang.interface.mixins.subject_mixin import SubjectMixin
from optlang.interface.mixins.symbolic_mixin import SymbolicMixin

__all__ = ("BoundsMixin",)

LOGGER = logging.getLogger(__name__)


class BoundsMixin(SymbolicMixin, SubjectMixin):
    """
    Provide bounds properties to an inheriting class.

    Also depends on there being an observer for proper functioning.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance from all the mixin classes, the ``__slots__``
    attribute is defined to be empty. A child class making use of the
    `BoundsMixin` is expected to define at least the following slots::

        __slots__ = ("_lb", "_numeric_lb", "_ub", "_numeric_ub")

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super(BoundsMixin, self).__init__(**kwargs)
        self._lb = None
        self._numeric_lb = None
        self._ub = None
        self._numeric_ub = None

    @staticmethod
    def _check_bounds(lb, ub):
        LOGGER.debug("Comparing %s with %s.", str(lb), str(ub))
        if lb is None or ub is None:
            return
        if lb > ub:
            raise ValueError(
                "The lower bound must be less than or equal to the upper bound "
                "({} <= {}).".format(lb, ub))

    @property
    def lb(self):
        """Modify the lower bound."""
        return self._lb

    @lb.setter
    def lb(self, value):
        # Conversion to a numeric value is done at this place because if it
        # fails the attributes are not yet changed.
        numeric = self._evaluate(value)
        self._set_numeric_lb(numeric)
        self._disregard_symbols(self.lb, "lb")
        self._observe_symbols(value, "lb")
        self._lb = value

    def _set_numeric_lb(self, value):
        self._check_bounds(value, self._numeric_ub)
        self._numeric_lb = value
        try:
            self._observer.update_lb(self, value)
        except (AttributeError, ReferenceError):
            # Observer is not set or no longer exists.
            pass

    @property
    def ub(self):
        """Modify the upper bound."""
        return self._ub

    @ub.setter
    def ub(self, value):
        # Conversion to a numeric value is done at this place because if it
        # fails the attributes are not yet changed.
        numeric = self._evaluate(value)
        self._set_numeric_ub(numeric)
        self._disregard_symbols(self.ub, "ub")
        self._observe_symbols(value, "ub")
        self._ub = value

    def _set_numeric_ub(self, value):
        self._check_bounds(self._numeric_lb, value)
        self._numeric_ub = value
        try:
            self._observer.update_ub(self, value)
        except (AttributeError, ReferenceError):
            # Observer is not set or no longer exists.
            pass

    @property
    def bounds(self):
        """Modify the lower and upper bound."""
        return self._lb, self._ub

    @bounds.setter
    def bounds(self, pair):
        lb, ub = pair
        # Conversion to a numeric value is done at this place because if it
        # fails the attributes are not yet changed.
        num_lb = self._evaluate(lb)
        num_ub = self._evaluate(ub)
        self._set_numeric_bounds(num_lb, num_ub)
        self._disregard_symbols(self.lb, "bounds")
        self._disregard_symbols(self.ub, "bounds")
        self._observe_symbols(lb, "bounds")
        self._observe_symbols(ub, "bounds")
        self._lb = lb
        self._ub = ub

    def _set_numeric_bounds(self, lower, upper):
        self._check_bounds(lower, upper)
        self._numeric_lb = lower
        self._numeric_ub = upper
        try:
            self._observer.update_bounds(self, lower, upper)
        except (AttributeError, ReferenceError):
            # Observer is not set or no longer exists.
            pass
