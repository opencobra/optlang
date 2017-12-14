# -*- coding: utf-8 -*-

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

from __future__ import absolute_import

import io
import sys

import versioneer
from setuptools import setup


setup_requirements = []
# prevent pytest-runner from being installed on every invocation
if set(['pytest', 'test', 'ptr']).intersection(sys.argv):
    setup_requirements.append("pytest-runner")

with io.open('requirements.txt') as file_handle:
    requirements = file_handle.readlines()

with io.open('test_requirements.txt') as file_handle:
    test_requirements = file_handle.readlines()


# All other keys are defined in setup.cfg under [metadata] and [options].
setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    setup_requires=setup_requirements,
    install_requires=requirements,
    tests_require=test_requirements
)
