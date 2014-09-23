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
import unittest
import urllib2
import nose

try:

    import cplex

    from optlang.cplex_interface import Model

    #problems from http://miplib.zib.de/miplib2003/miplib2003.php

    MULTISPACE_RE = re.compile("\s+")
    MIPLIB_URL = "http://miplib.zib.de/miplib2003/download/%s"

    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
    SOLUTION = os.path.join(DATA_PATH, "miplib2003.txt")
    PROBLEMS_DIR = os.path.join(DATA_PATH, "miplib2003")

    SOLUTION_MAPPING = {
        "=opt=": "optimal",
        "=inf=": "infeasible"
    }

    def download_file(file_name, problem_dir):
        _in = urllib2.urlopen(MIPLIB_URL % file_name)
        with open(os.path.join(problem_dir, file_name), 'w') as out:
            out.write(_in.read())
        _in.close()

    def load_problem(mps_file,):
        prob_tmp_file = tempfile.mktemp(suffix='.mps')
        with open(prob_tmp_file, 'w') as tmp_handle:
            f = gzip.open(mps_file, 'rb')
            tmp_handle.write(f.read())
            f.close()

        problem = cplex.Cplex()
        problem.read(prob_tmp_file)
        model = Model(problem=problem)
        model.configuration.presolve = True
        model.configuration.timeout = 60 * 10
        return problem, model

    def check_dimensions(model, cplex_problem):
        nose.tools.assert_true(cplex_problem.variables.get_num() == len(model.variables))

    def check_optimization(model, expected_solution):
        status = model.optimize()
        if not status is "time_limit":
            nose.tools.assert_equals(status, expected_solution.sol_type)

            if status is "optimal":
                nose.tools.assert_almost_equal(expected_solution.f, model.objective.value, places=4)

                # for v in model.variables:
                #     nose.tools.assert_almost_equal(expected_solution.solution[v.name], v.primal, places=4)


    class MockSolution(object):
        def __init__(self, name, sol_type, f, problem_dir):
            self.name = name
            self.sol_type = sol_type
            self.f = f
            self.problem_dir = problem_dir
            self._solution = None

        @property
        def solution(self):
            if self._solution is None:
                self._solution = {}
                solution_file = os.path.join(self.problem_dir, "%s.sol.gz" % self.name)
                try:
                    if not os.path.exists(solution_file):
                        download_file("%s.sol.gz" % self.name, self.problem_dir)
                    f = gzip.open(solution_file, 'rb')
                    for line in f:
                        data = MULTISPACE_RE.split(line)
                        self._solution[data[1]] = float(data[2])
                except Exception as e:
                    raise e

            return self._solution

    def test_miplib(solutions=SOLUTION, problem_dir=PROBLEMS_DIR):
        with open(solutions, "rU") as file_handler:
            for line in file_handler:
                data = MULTISPACE_RE.split(line)
                sol_type, name, f = data[0], data[1], float(data[2])
                sol_type = SOLUTION_MAPPING[sol_type]
                expected_solution = MockSolution(name, sol_type, f, problem_dir)
                problem_file = os.path.join(problem_dir, "%s.mps.gz" % name)
                try:
                    if not os.path.exists(problem_file):
                        download_file("%s.mps.gz" % name, problem_dir)

                except Exception as e:
                    print e
                    continue

                cplex_problem, model = load_problem(problem_file)
                func = partial(check_dimensions, model, cplex_problem)
                func.description = "test_miplib_dimensions_%s (%s)" % (name, os.path.basename(str(__file__)))
                yield func

                func = partial(check_optimization, model, expected_solution)
                func.description = "test_miplib_optimization_%s (%s)" % (name, os.path.basename(str(__file__)))
                yield func

except ImportError, e:

    if e.message.find('cplex') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + e.message)
            def test_fail(self):
                pass
    else:
        raise


if __name__ == '__main__':
        nose.runmodule()