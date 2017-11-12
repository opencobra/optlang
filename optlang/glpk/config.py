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

import swiglpk as glpk
from six import add_metaclass, iteritems

import optlang.interface as interface
from optlang.interface import MathematicalProgrammingConfiguration
from optlang.util import inheritdocstring

GLPK_STATUS_TO_STATUS = {
    glpk.GLP_UNDEF: interface.UNDEFINED,
    glpk.GLP_FEAS: interface.FEASIBLE,
    glpk.GLP_INFEAS: interface.INFEASIBLE,
    glpk.GLP_NOFEAS: interface.INFEASIBLE,
    glpk.GLP_OPT: interface.OPTIMAL,
    glpk.GLP_UNBND: interface.UNBOUNDED
}

GLPK_VTYPE_TO_VTYPE = {
    glpk.GLP_CV: 'continuous',
    glpk.GLP_IV: 'integer',
    glpk.GLP_BV: 'binary'
}

VTYPE_TO_GLPK_VTYPE = {
    val: key for key, val in iteritems(GLPK_VTYPE_TO_VTYPE)}


@add_metaclass(inheritdocstring)
class Configuration(MathematicalProgrammingConfiguration):
    def __init__(self, presolve="auto", verbosity=0, timeout=None, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self._smcp = glpk.glp_smcp()
        self._iocp = glpk.glp_iocp()
        glpk.glp_init_smcp(self._smcp)
        glpk.glp_init_iocp(self._iocp)
        self._max_time = min(self._smcp.tm_lim, self._iocp.tm_lim)
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout

    def __getstate__(self):
        return {'presolve': self.presolve, 'verbosity': self.verbosity, 'timeout': self.timeout}

    def __setstate__(self, state):
        self.__init__()
        for key, val in iteritems(state):
            setattr(self, key, val)

    def _set_presolve(self, value):
        self._smcp.presolve = {
            False: glpk.GLP_OFF, True: glpk.GLP_ON, "auto": glpk.GLP_OFF}[value]
        self._iocp.presolve = {
            False: glpk.GLP_OFF, True: glpk.GLP_ON, "auto": glpk.GLP_OFF}[value]

    def _set_verbosity(self, value):
        if value == 0:
            glpk.glp_term_out(glpk.GLP_OFF)
            self._smcp.msg_lev = glpk.GLP_MSG_OFF
            self._iocp.msg_lev = glpk.GLP_MSG_OFF
        elif value == 1:
            glpk.glp_term_out(glpk.GLP_OFF)
            self._smcp.msg_lev = glpk.GLP_MSG_ERR
            self._iocp.msg_lev = glpk.GLP_MSG_ERR
        elif value == 2:
            glpk.glp_term_out(glpk.GLP_OFF)
            self._smcp.msg_lev = glpk.GLP_MSG_ON
            self._iocp.msg_lev = glpk.GLP_MSG_ON
        elif value == 3:
            glpk.glp_term_out(glpk.GLP_ON)
            self._smcp.msg_lev = glpk.GLP_MSG_ALL
            self._iocp.msg_lev = glpk.GLP_MSG_ALL
        else:
            raise ValueError(
                "%s is not a valid verbosity level ranging between 0 and 3."
                % value
            )

    def _set_timeout(self, value):
        if value is None:
            self._smcp.tm_lim = self._max_time
            self._iocp.tm_lim = self._max_time
        else:
            self._smcp.tm_lim = value * 1000  # milliseconds to seconds
            self._iocp.tm_lim = value * 1000

    def _tolerance_functions(self):
        return {
            "feasibility": (
                lambda: self._smcp.tol_bnd,
                lambda x: setattr(self._smcp, 'tol_bnd', x)
            ),
            "optimality": (
                lambda: self._iocp.tol_obj,
                lambda x: setattr(self._iocp, 'tol_obj', x)
            ),
            "integrality": (
                lambda: self._iocp.tol_int,
                lambda x: setattr(self._iocp, 'tol_int', x)
            )
        }

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        self._set_presolve(value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._set_verbosity(value)
        self._verbosity = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._set_timeout(value)
        self._timeout = value

