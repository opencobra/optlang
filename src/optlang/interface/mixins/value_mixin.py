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

from optlang.interface.mixins.solver_state_mixin import SolverStateMixin

__all__ = ("ValueMixin",)


class ValueMixin(SolverStateMixin):
    """
    Provide primal and dual value properties to a child class.

    Also depends on there being a solver state reference for proper functioning.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance from all the mixin classes, the ``__slots__``
    attribute is defined to be empty. A child class making use of the
    `ValueMixin` is expected to define at least the following slots::

        __slots__ = ()

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super(ValueMixin, self).__init__(**kwargs)

    @property
    def primal(self):
        """Return the primal value (None if no solution exists)."""
        try:
            return self._solver.get_primal(self)
        except (AttributeError, ReferenceError):
            # Observable is not set or no longer exists.
            return None

    @property
    def dual(self):
        """Return the dual value (None if no solution exists)."""
        try:
            return self._solver.get_dual(self)
        except (AttributeError, ReferenceError):
            # Observable is not set or no longer exists.
            return None
