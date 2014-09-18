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

from functools import partial
import gzip

import os
import re
import tempfile
import urllib2
import nose
from optlang.glpk_interface import Model
from swiglpk import glp_read_mps, GLP_MPS_FILE, glp_create_prob, glp_get_num_cols, GLP_BR_PCH

MULTISPACE_RE = re.compile("\s+")
MIPLIB_URL = "http://miplib.zib.de/download/%s"

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
SOLUTION = os.path.join(DATA_PATH, "miplib_benchmark.solutions")
PROBLEMS_DIR = os.path.join(DATA_PATH, "miplib2010")

SOLUTION_MAPPING = {
    "=opt=": "optimal",
    "=inf=": "infeasible"
}


def download_file(file_name, problem_dir):
    _in = urllib2.urlopen(MIPLIB_URL % file_name)
    with open(os.path.join(problem_dir, file_name), 'w') as out:
        out.write(_in.read())
    _in.close()


def load_problem(mps_file):
    tmp_file = tempfile.mktemp(suffix='.mps')
    with open(tmp_file, 'w') as tmp_handle:
        f = gzip.open(mps_file, 'rb')
        tmp_handle.write(f.read())
        f.close()

    problem = glp_create_prob()
    glp_read_mps(problem, GLP_MPS_FILE, None, tmp_file)
    model = Model(problem=problem)
    model.configuration.presolve = True
    model.configuration.verbosity = 3
    model.configuration.timeout = 60 * 10
    model.configuration._iocp.br_tech = GLP_BR_PCH
    return problem, model


def check_dimensions(model, glpk_problem):
    nose.tools.assert_true(glp_get_num_cols(glpk_problem) == len(model.variables))


def check_optimization(status, model, expected_solution):
    nose.tools.assert_equals(status, expected_solution.sol_type)
    if status is "optimal":
        nose.tools.assert_almost_equal(expected_solution.f, model.objective.value, places=4)


def check_variables(model, expected_solution):
    for v in model.variables:
        nose.tools.assert_almost_equal(expected_solution.solutions[v.id], v.primal, places=4)


class MockupSolution():
    def __init__(self, name, sol_type, f, problem_dir):
        self.name = name
        self.sol_type = sol_type
        self.f = f
        self.problems_dir = problem_dir
        self._solution = None

    @property
    def solution(self):
        if self._solution is None:
            self._solution = {}
            solution_file = os.path.join(self.problem_dir, "%s.sol.gz" % self.name)
            try:
                if not os.path.exists(solution_file):
                    download_file("%s.sol.gz" % self.name, self.problems_dir)
                f = gzip.open(solution_file, 'rb')
                for line in f:
                    data = MULTISPACE_RE.split(line)
                    self._solution[data[0]] = float(data[1])
            except:
                pass

        return self._solution


def test_miplib(solutions=SOLUTION, problem_dir=PROBLEMS_DIR):
    with open(solutions, "rU") as fhandler:
        for line in fhandler:
            data = MULTISPACE_RE.split(line)
            if len(data) == 3:
                sol_type, name, f = data[0], data[1], data[2]
            else:
                sol_type, name, f = data[0], data[1], None
            if name == "air04":
                sol_type = SOLUTION_MAPPING[sol_type]
                print sol_type
                expected_solution = MockupSolution(name, sol_type, f, problem_dir)
                problem_file = os.path.join(problem_dir, "%s.mps.gz" % name)
                try:
                    if not os.path.exists(problem_file):
                        download_file("%s.mps.gz" % name, problem_dir)
                except:
                    continue

                glpk_problem, model = load_problem(problem_file)
                status = model.optimize()
                func = partial(check_dimensions, model, glpk_problem)
                func.description = "test_miplib_check_dimensions_%s (%s)" % (name, os.path.basename(str(__file__)))
                yield func
                if status == "timeout":
                    continue

                func = partial(check_optimization, status, model, expected_solution)
                func.description = "test_miplib_check_optimization_%s (%s)" % (name, os.path.basename(str(__file__)))
                yield func

                if expected_solution.sol_type == "optimal":
                    func = partial(check_variables, model, expected_solution)
                    func.description = "test_miplib_check_variables_%s (%s)" % (name, os.path.basename(str(__file__)))
                    yield func


if __name__ == '__main__':
    nose.runmodule()