# Copyright 2014 Novo Nordisk Foundation Center for Biosustainability, DTU.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import gzip
import os
import tempfile
from functools import partial

import nose
from swiglpk import glp_read_mps, GLP_MPS_FILE, glp_create_prob, glp_get_num_cols

from optlang.glpk_interface import Model

# problems from http://miplib.zib.de/miplib2003/miplib2003.php

TRAVIS = os.getenv('TRAVIS', False)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
SOLUTION = os.path.join(DATA_PATH, "miplib2003.json")
PROBLEMS_DIR = os.path.join(DATA_PATH, "miplib2003")


def load_problem(mps_file):
    prob_tmp_file = tempfile.mktemp(suffix='.mps')
    with open(prob_tmp_file, 'wb') as tmp_handle:
        f = gzip.open(mps_file, 'rb')
        tmp_handle.write(f.read())
        f.close()

    problem = glp_create_prob()
    glp_read_mps(problem, GLP_MPS_FILE, None, prob_tmp_file)
    model = Model(problem=problem)
    model.configuration.presolve = True
    model.configuration.verbosity = 3
    model.configuration.timeout = 60 * 9
    return problem, model


def check_dimensions(model, glpk_problem):
    nose.tools.assert_true(glp_get_num_cols(glpk_problem) == len(model.variables))


def check_optimization(model, expected_solution):
    status = model.optimize()
    if status is not "time_limit":
        nose.tools.assert_equals(status, expected_solution['status'])

        if status is "optimal":
            nose.tools.assert_almost_equal(expected_solution['solution'], model.objective.value, places=4)


def test_miplib(solutions=SOLUTION, problem_dir=PROBLEMS_DIR):
    if TRAVIS:
        raise nose.SkipTest('Skipping extensive MILP tests on travis-ci.')
    with open(solutions, "r") as f:
        data = json.load(f)
        print(data)
    for name, problem_data in data.items():
        problem_file = os.path.join(problem_dir, "{}.mps.gz".format(name))

        glpk_problem, model = load_problem(problem_file)
        func = partial(check_dimensions, model, glpk_problem)
        func.description = "test_miplib_dimensions_%s (%s)" % (name, os.path.basename(str(__file__)))
        yield func

        func = partial(check_optimization, model, problem_data)
        func.description = "test_miplib_optimization_%s (%s)" % (name, os.path.basename(str(__file__)))
        yield func


if __name__ == '__main__':
    nose.runmodule()
