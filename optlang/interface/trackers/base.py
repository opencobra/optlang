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

__all__ = ("BaseChangeTracker",)

LOGGER = logging.getLogger(__name__)


class BaseChangeTracker(object):

    def __init__(self, **kwargs):
        super(BaseChangeTracker, self).__init__(**kwargs)
        self._to_add = list()
        self._to_rm = list()

    @staticmethod
    def _iter_unique(iterable):
        seen = set()
        for obj in reversed(iterable):
            if obj in seen:
                continue
            yield obj
            seen.add(obj)
        iterable.clear()

    @staticmethod
    def _iter_last_unique(iterable):
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
        return self._iter_unique(self._to_add)

    def remove(self, obj):
        LOGGER.debug("Tracked removal of '%s'.", obj.name)
        self._to_rm.append(obj)

    def iter_to_remove(self):
        return self._iter_unique(self._to_rm)
