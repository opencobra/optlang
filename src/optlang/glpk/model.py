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

from optlang.interface import Model, parse_optimization_expression
# from optlang.util import inheritdocstring
from optlang.glpk.config import Configuration, VTYPE_TO_GLPK_VTYPE

__all__ = ("GLPKModel",)

LOGGER = logging.getLogger(__name__)
GLPK_DIRECTION = {
    "min": glpk.GLP_MIN,
    "max": glpk.GLP_MAX
}

# @add_metaclass(inheritdocstring)
class GLPKModel(Model):
    def __init__(self, problem=None, **kwargs):

        super(GLPKModel, self).__init__(**kwargs)

        self.configuration = Configuration()
        self._objective_offset = 0

        if problem is None:
            self._problem = glpk.glp_create_prob()
            glpk.glp_create_index(self._problem)
            if self.name is not None:
                glpk.glp_set_prob_name(self._problem, str(self.name))
        else:
            raise NotImplementedError()

    objective = property(Model.objective)

    @objective.setter
    def objective(self, new):
        for var in self._objective.variables:
            idx = glpk.glp_find_col(self._problem, var.name)
            glpk.glp_set_obj_coef(self._problem, idx, 0.)
        super(GLPKModel, self.__class__).objective.fset(self, new)

        offset, coef_dict, _ = parse_optimization_expression(new, linear=True)
        self._objective_offset = offset

        for var, coef in coef_dict.items():
            idx = glpk.glp_find_col(self._problem, var.name)
            glpk.glp_set_obj_coef(self._problem, idx, coef)

        glpk.glp_set_obj_dir(self._problem,
                             GLPK_DIRECTION[self._objective.direction])

    def _add_variables(self, variables):
        variables = list(variables)
        start = glpk.glp_get_num_cols(self._problem)
        glpk.glp_add_cols(self._problem, len(variables))

        for idx, var in enumerate(variables, start=start + 1):
            glpk.glp_set_col_name(self._problem, idx, var.name)
            glpk.glp_set_col_kind(
                self._problem, idx, VTYPE_TO_GLPK_VTYPE[var.type])
            self._set_variable_bound(var, idx)
        super(GLPKModel, self)._add_variables(variables)

    def _set_variable_bounds(self, variables):
        pass

    def _set_variable_bound(self, variable, index=None):
        if index is None:
            index = glpk.glp_find_col(self._problem, variable.name)
        if variable.lb is None and variable.ub is None:
            glpk.glp_set_col_bnds(self._problem, index, glpk.GLP_FR, 0.0, 0.0)
        elif variable.lb is None:
            glpk.glp_set_col_bnds(self._problem, index, glpk.GLP_UP,
                                  0.0, variable.ub)
        elif variable.ub is None:
            glpk.glp_set_col_bnds(self._problem, index, glpk.GLP_LO,
                                  variable.lb, 0.0)
        elif variable.lb == variable.ub:
            glpk.glp_set_col_bnds(self._problem, index, glpk.GLP_FX,
                                  variable.lb, 0.0)
        elif variable.lb < variable.ub:
            glpk.glp_set_col_bnds(self._problem, index, glpk.GLP_DB,
                                  variable.lb, variable.ub)
        else:
            raise ValueError(
                "Unknown bound types or incomparable bounds {} and {} of "
                "variable {}.".format(variable.lb, variable.ub, variable.name))

    def _add_constraints(self, constraints, sloppy=False):
        constraints = list(constraints)
        start = glpk.glp_get_num_rows(self._problem)
        glpk.glp_add_rows(self._problem, len(constraints))
        for idx, constr in enumerate(constraints, start=start + 1):
            glpk.glp_set_row_name(self._problem, idx, constr.name)
            self._set_constraint_bound(constr, idx)
        super(GLPKModel, self)._add_constraints(constraints, sloppy=sloppy)

    def _set_constraint_bound(self, constraint, index=None):
        if index is None:
            index = glpk.glp_find_row(self._problem, constraint.name)
        if constraint.lb is None and constraint.ub is None:
            glpk.glp_set_row_bnds(self._problem, index, glpk.GLP_FR, 0.0, 0.0)
        elif constraint.lb is None:
            glpk.glp_set_row_bnds(self._problem, index, glpk.GLP_UP,
                                  0.0, constraint.ub)
        elif constraint.ub is None:
            glpk.glp_set_row_bnds(self._problem, index, glpk.GLP_LO,
                                  constraint.lb, 0.0)
        elif constraint.lb == constraint.ub:
            glpk.glp_set_row_bnds(self._problem, index, glpk.GLP_FX,
                                  constraint.lb, 0.0)
        elif constraint.lb < constraint.ub:
            glpk.glp_set_row_bnds(self._problem, index, glpk.GLP_DB,
                                  constraint.lb, constraint.ub)
        else:
            raise ValueError(
                "Unknown bound types or incomparable bounds {} and {} of "
                "constraint {}.".format(constraint.lb, constraint.ub,
                                        constraint.name))

    # def update_variable_lb(self, var, value):
    #     col = glpk.glp_find_col(self._problem, var.name)
    #     LOGGER.debug("Updating the lower bound of var %s in column %d.",
    #                  var.name, col)
        # glpk.glp_set_col_bnds(self._problem, col)
