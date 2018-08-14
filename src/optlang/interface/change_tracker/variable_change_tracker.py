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

from optlang.interface.change_tracker.name_change_tracker import NameChangeTracker
from optlang.interface.change_tracker.bounds_change_tracker import BoundsChangeTracker

__all__ = ("VariableChangeTracker",)

LOGGER = logging.getLogger(__name__)


class VariableChangeTracker(BoundsChangeTracker, NameChangeTracker):

    def __init__(self, **kwargs):
        super(VariableChangeTracker, self).__init__(**kwargs)
        self._type = list()

    def update_type(self, obj, kind):
        LOGGER.debug("Tracked type update to '%s'.", kind)
        self._type.append((obj, kind))

    def iter_type(self):
        return self._iter_last_unique(self._type)
