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

from optlang.interface.change_tracker.base_change_tracker import BaseChangeTracker

__all__ = ("BoundsChangeTracker",)

LOGGER = logging.getLogger(__name__)


class BoundsChangeTracker(BaseChangeTracker):

    def __init__(self, **kwargs):
        super(BoundsChangeTracker, self).__init__(**kwargs)
        self._lb = []
        self._ub = []
        self._bounds = []

    def update_lb(self, obj, value):
        LOGGER.debug("Tracked lower bound update to %s.", str(value))
        self._lb.append((obj, value))

    def iter_lb(self):
        return self._iter_last_unique_obj(self._lb)

    def update_ub(self, obj, value):
        LOGGER.debug("Tracked upper bound update to %s.", str(value))
        self._ub.append((obj, value))

    def iter_ub(self):
        return self._iter_last_unique_obj(self._ub)

    def update_bounds(self, obj, lb, ub):
        LOGGER.debug("Tracked bounds update to %s, %s.", str(lb), str(ub))
        self._bounds.append((obj, lb, ub))

    def iter_bounds(self):
        return self._iter_last_unique_obj(self._bounds)
