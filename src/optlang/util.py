# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
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


"""Provide utility functions for optlang."""

import logging

LOGGER = logging.getLogger(__name__)


def is_numeric(obj):
    if isinstance(obj, (int, float)) or getattr(obj, "is_Number", False):
        return True
    else:
        try:
            float(obj)
        except ValueError:
            return False
        else:
            return True
