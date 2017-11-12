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

import inspect

from optlang.util import SolverTolerances

__all__ = (
    "MathematicalProgrammingConfiguration",
    "EvolutionaryOptimizationConfiguration"
)


class Configuration(object):
    """
    Optimization solver configuration.
    This object allows the user to change certain parameters and settings in the solver.
    It is meant to allow easy access to a few common and important parameters. For information on changing
    other solver parameters, please consult the documentation from the solver provider.
    Some changeable parameters are listed below. Note that some solvers might not implement all of these
    and might also implement additional parameters.

    Attributes
    ----------
    verbosity: int from 0 to 3
        Changes the level of output.
    timeout: int or None
        The time limit in second the solver will use to optimize the problem.
    presolve: Boolean or 'auto'
        Tells the solver whether to use (solver-specific) pre-processing to simplify the problem.
        This can decrease solution time, but also introduces overhead. If set to 'auto' the solver will
        first try to solve without pre-processing, and only turn in on in case no optimal solution can be found.
    lp_method: str
        Select which algorithm the LP solver uses, e.g. simplex, barrier, etc.

    """

    @classmethod
    def clone(cls, config, problem=None, **kwargs):
        properties = (k for k, v in inspect.getmembers(cls, predicate=inspect.isdatadescriptor) if
                      not k.startswith('__'))
        parameters = {property: getattr(config, property) for property in properties if hasattr(config, property)}
        return cls(problem=problem, **parameters)

    def __init__(self, problem=None, *args, **kwargs):
        self.problem = problem
        self._add_tolerances()

    @property
    def verbosity(self):
        """Verbosity level.

        0: no output
        1: error and warning messages only
        2: normal output
        3: full output
        """
        raise NotImplementedError

    @verbosity.setter
    def verbosity(self, value):
        raise NotImplementedError

    @property
    def timeout(self):
        """Timeout parameter (seconds)."""
        raise NotImplementedError

    @timeout.setter
    def timeout(self):
        raise NotImplementedError

    @property
    def presolve(self):
        """
        Turn pre-processing on or off. Set to 'auto' to only use presolve if no optimal solution can be found.
        """
        raise NotImplementedError

    @presolve.setter
    def presolve(self):
        raise NotImplementedError

    def _add_tolerances(self):
        self.tolerances = SolverTolerances(self._tolerance_functions())

    def _tolerance_functions(self):
        """
        This should be implemented in child classes. Must return a dict, where keys are available tolerance parameters
        and values are tuples of (getter_function, setter_function).
        The getter functions must be callable with no arguments and the setter functions must be callable with the
        new value as the only argument
        """
        return {}

    def __setstate__(self, state):
        self.__init__()


class MathematicalProgrammingConfiguration(Configuration):
    def __init__(self, *args, **kwargs):
        super(MathematicalProgrammingConfiguration, self).__init__(*args, **kwargs)

    @property
    def presolve(self):
        """If the presolver should be used (if available)."""
        raise NotImplementedError

    @presolve.setter
    def presolve(self, value):
        raise NotImplementedError


class EvolutionaryOptimizationConfiguration(Configuration):
    """docstring for HeuristicOptimization"""

    def __init__(self, *args, **kwargs):
        super(EvolutionaryOptimizationConfiguration, self).__init__(*args, **kwargs)

