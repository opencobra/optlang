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
import pytest

from swiglpk import glp_read_mps, GLP_MPS_FILE, glp_create_prob, glp_get_num_cols

from optlang.glpk_interface import Model

# problems from http://miplib.zib.de/miplib2003/miplib2003.php

CI = os.getenv('CI', False)

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
    assert glp_get_num_cols(glpk_problem) == len(model.variables)


def check_optimization(model, expected_solution):
    status = model.optimize()
    if status != "time_limit":
        assert status == expected_solution['status']

        if status == "optimal":
            assert expected_solution['solution'] == pytest.approx(model.objective.value, 1e-4, 1e-4)


with open(SOLUTION, "r") as f:
    data = json.load(f)
    print(data)


@pytest.mark.skipif(CI, reason="too slow on CI")
@pytest.mark.parametrize("problem", data)
def test_miplib(problem):
    problem_file = os.path.join(PROBLEMS_DIR, "{}.mps.gz".format(problem))

    glpk_problem, model = load_problem(problem_file)
    check_dimensions(model, glpk_problem)

    check_optimization(model, data[problem])
