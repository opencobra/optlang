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

__all__ = ("ObserverMixin",)


class ObserverMixin(object):
    """
    Provide a method to set an observer on an inheriting class.

    Notes
    -----
    Trying to access methods of the observer may raise an
    `AttributeError` if it is not set (`None`) or a `ReferenceError` if
    the observable no longer exists.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance from all the mixin classes, the ``__slots__``
    attribute is defined to be empty. A child class making use of the
    `ObserverMixin` is expected to define at least the following slots::

        __slots__ = ("_observer",)

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super(ObserverMixin, self).__init__(**kwargs)
        self._observer = None

    def set_observer(self, observer):
        """Set the instance's observer."""
        self._observer = proxy(observer)

    def unset_observer(self):
        """Unset the instance's observer."""
        self._observer = None
