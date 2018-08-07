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

from optlang.interface.mixins.observer_mixin import ObserverMixin

__all__ = ("NameMixin",)


class NameMixin(ObserverMixin):
    """
    Provide a name property to an inheriting class.

    Also depends on there being an observer for proper functioning.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance from all the mixin classes, the ``__slots__``
    attribute is defined to be empty. A child class making use of the
    `NameMixin` is expected to define at least the following slots::

        __slots__ = ("_name",)

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super(NameMixin, self).__init__(**kwargs)
        self._name = None

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        # Ensure that name is str and not binary or unicode.
        # Some solvers only support the `str` type in Python 2.
        if PY2:
            name = str(name)
        if len(name) == 0:
            raise ValueError("The name must not be empty.")
        if any(char.isspace() for char in name):
            raise ValueError(
                "The name cannot contain whitespace characters.")
        self._name = name
        # We have to access the observer in this way because sympy uses slots
        #  with a ``name`` attribute during instantiation and thus accesses
        # this property before ``_observer`` exists.
        observer = getattr(self, "_observer", None)
        try:
            observer.update_name(self, name)
        except (AttributeError, ReferenceError):
            # Observer is not set or no longer exists.
            pass


