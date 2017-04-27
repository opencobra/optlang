# Copyright 2015 Novo Nordisk Foundation Center for Biosustainability,
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


class SolverError(Exception):
    """Reraise solver specific errors with this unified optlang error instead."""

    def __init__(self, message, **kwargs):
        super(SolverError, self).__init__(message, **kwargs)


class ContainerAlreadyContains(Exception):
    """
    This exception is raised when the name of an object being added to a Container is already
    taken by another object.
    """
    def __init__(self, message):
        super(ContainerAlreadyContains, self).__init__(message)


class IndicatorConstraintsNotSupported(Exception):
    """
    This exception is raised when trying to add indicator variables to a constraint using a solver
    that does not support indicator variables.
    """
    def __init__(self, message):
        super(IndicatorConstraintsNotSupported, self).__init__(message)
