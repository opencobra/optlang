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

from weakref import proxy

__all__ = ("SolverStateMixin",)


class SolverStateMixin(object):
    """
    Provide an interface for setting a weak reference to solver state.

    At the moment only reference can be set at a time. It would be possible
    to associate an object with multiple solvers and refer to all their
    states for this instance. We decided against it because it will likely do
    more harm than good.

    Notes
    -----
    Trying to access methods of the solver state object may raise an
    `AttributeError` if it is not set (`None`) or a `ReferenceError` if
    the object no longer exists.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance from all the mixin classes, the ``__slots__``
    attribute is defined to be empty. A child class making use of the
    `SolverStateMixin` is expected to define at least the following slots::

        __slots__ = ("_solver",)

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super(SolverStateMixin, self).__init__(**kwargs)
        self._solver = None

    def set_solver(self, solver):
        """Set a reference to the solver state."""
        self._solver = proxy(solver)

    def unset_solver(self):
        """Unset the solver state reference."""
        self._solver = None
