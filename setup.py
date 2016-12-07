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

import os

from setuptools import setup, find_packages
from itertools import chain

import versioneer

versioneer.VCS = 'git'
versioneer.versionfile_source = 'optlang/_version.py'
versioneer.versionfile_build = 'optlang/_version.py'
versioneer.tag_prefix = ''  # tags are like 1.2.0
versioneer.parentdir_prefix = 'optlang-'  # dirname like 'myproject-1.2.0'

# Run
# pandoc --from=markdown --to=rst README.md -o README.rst
# from time to time, to keep README.rst updated
with open('README.rst', 'r') as f:
    description = f.read()

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requirements = []
else:
    # requirements = ['sympy>=1.0.0', 'six>=1.9.0']
    with open("requirements.txt") as infile:
        requirements = infile.read().splitlines()


extra_requirements = {
    'test': ['nose>=1.3.7', 'rednose>=0.4.3', 'coverage>=4.0.3', 'jsonschema>=2.5'],
}
extra_requirements['all'] = list(set(chain(*extra_requirements.values())))

setup(
    name='optlang',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=requirements,  # from requirements.txt
    extras_require=extra_requirements,
    test_suite='nose.collector',
    author='Nikolaus Sonnenschein',
    author_email='niko.sonnenschein@gmail.com',
    description='Formulate optimization problems using sympy expressions and solve them using interfaces to third-party optimization software (e.g. GLPK).',
    license='Apache License Version 2.0',
    url='https://github.com/biosustain/optlang',
    long_description=description,
    keywords=['optimization', 'sympy', 'mathematical programming', 'heuristic optimization'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: Apache Software License',
    ],
)
