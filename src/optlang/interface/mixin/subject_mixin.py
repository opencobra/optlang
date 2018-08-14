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

__all__ = ("SubjectMixin",)


class SubjectMixin(object):
    """
    Turn an inheriting class into an solver subject.

    An instance can be subscribed to and it will publish updates to the
    observing object. It can also be unsubscribed from. It thus follows the
    common Observer design pattern [1]_.

    In a variation to the common pattern, only one observer at a time is
    allowed. (This could fairly easily be changed in future but is probably
    unwanted in the context of variables, constraints, and solvers.)

    Notes
    -----
    Trying to access methods of the observer may raise an
    `AttributeError` if it is not set (`None`) or a `ReferenceError` if
    the solver no longer exists.

    Warnings
    --------
    As described in the `mixins` package documentation, in order to enable
    multiple inheritance from all the mixin classes, the ``__slots__``
    attribute is defined to be empty. A child class making use of the
    `SubjectMixin` is expected to define at least the following slots::

        __slots__ = ("_observer",)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Observer_pattern

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super(SubjectMixin, self).__init__(**kwargs)
        self._observer = None

    def subscribe(self, observer):
        """Set an observer on the instance."""
        self._observer = proxy(observer)

    def unsubscribe(self):
        """Unset the observer from the instance."""
        self._observer = None
