# -*- coding: utf-8 -*-

# Copyright 2013-2017 Novo Nordisk Foundation Center for Biosustainability,
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

import swiglpk as glpk
from six import add_metaclass

from optlang.interface import Model
from optlang.util import inheritdocstring
from optlang.glpk.config import Configuration, VTYPE_TO_GLPK_VTYPE

__all__ = ("GLPKModel",)

LOGGER = logging.getLogger(__name__)


@add_metaclass(inheritdocstring)
class GLPKModel(Model):
    def __init__(self, problem=None, **kwargs):

        super(GLPKModel, self).__init__(**kwargs)

        self.configuration = Configuration()

        if problem is None:
            self._problem = glpk.glp_create_prob()
            glpk.glp_create_index(self._problem)
            if self.name is not None:
                glpk.glp_set_prob_name(self._problem, str(self.name))

    def add_variables(self, variables):
        start = glpk.glp_get_num_cols(self._problem)
        glpk.glp_add_cols(self._problem, len(variables))

        for idx, var in enumerate(variables, start=start + 1):
            glpk.glp_set_col_name(self._problem, idx, var.name)
            glpk.glp_set_col_kind(
                self._problem, idx, VTYPE_TO_GLPK_VTYPE[var.type])
        super(GLPKModel, self).add_variables(variables)

    def update_variable_lb(self, var, value):
        col = glpk.glp_find_col(self._problem, var.name)
        LOGGER.debug("Updating the lower bound of var %s in column %d.",
                     var.name, col)
        # glpk.glp_set_col_bnds(self._problem, col)
