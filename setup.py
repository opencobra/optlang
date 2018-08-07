#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
# Copyright 2018 Novo Nordisk Foundation Center for Biosustainability,
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

"""Setup the optlang package."""

from __future__ import absolute_import

import versioneer
from setuptools import setup


# All other arguments are defined in the `setup.cfg`.
setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    # Temporary workaround for https://github.com/pypa/setuptools/issues/1136.
    package_dir={"": "src"}
)
