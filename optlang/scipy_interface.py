# Copyright 2015 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scipy.optimize
from optlang import interface

class Model(interface.Model):
    """scipy.optimize.linprog interface"""

    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

    def optimize(self):
        raise NotImplementedError()
