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

__all__ = ("ChangeTracker", "VariableChangeTracker")

LOGGER = logging.getLogger(__name__)


class ChangeTracker(object):

    def __init__(self, **kwargs):
        super(ChangeTracker, self).__init__(**kwargs)
        self._to_add = list()
        self._to_rm = list()
        self._name = list()
        self._lb = list()
        self._ub = list()
        self._bounds = list()

    @staticmethod
    def _iter_last(iterable):
        seen = set()
        for pack in reversed(iterable):
            obj = pack[0]
            if obj in seen:
                continue
            yield pack
            seen.add(obj)
        iterable.clear()

    def add(self, obj):
        LOGGER.debug("Tracked addition of '%s'.", obj.name)
        self._to_add.append(obj)

    def iter_to_add(self):
        return self._iter_last(self._to_add)

    def remove(self, obj):
        LOGGER.debug("Tracked removal of '%s'.", obj.name)
        self._to_rm.append(obj)

    def iter_to_remove(self):
        return self._iter_last(self._to_rm)

    def update_name(self, obj, name):
        LOGGER.debug("Tracked name update to '%s'.", name)
        self._name.append((obj, name))

    def iter_name(self):
        return self._iter_last(self._name)

    def update_lb(self, obj, value):
        LOGGER.debug("Tracked lower bound update to %f.", value)
        self._lb.append((obj, value))

    def iter_lb(self):
        return self._iter_last(self._lb)

    def update_ub(self, obj, value):
        LOGGER.debug("Tracked upper bound update to %f.", value)
        self._ub.append((obj, value))

    def iter_ub(self):
        return self._iter_last(self._ub)

    def update_bounds(self, obj, lb, ub):
        LOGGER.debug("Tracked bounds update to %f, %f.", lb, ub)
        self._bounds.append((obj, lb, ub))

    def iter_bounds(self):
        return self._iter_last(self._bounds)


class VariableChangeTracker(ChangeTracker):
    def __init__(self, **kwargs):
        super(VariableChangeTracker, self).__init__(**kwargs)
        self._type = list()

    def update_type(self, obj, kind):
        LOGGER.debug("Tracked type update to '%s'.", kind)
        self._type.append((obj, kind))

    def iter_type(self):
        return self._iter_last(self._type)

