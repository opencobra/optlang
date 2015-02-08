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


"""Utility functions for optlang."""

import logging

import os


log = logging.getLogger(__name__)
import tempfile
from subprocess import check_output


def solve_with_glpsol(glp_prob):
    """Solve glpk problem with glpsol commandline solver. Mainly for testing purposes.

    # Examples
    # --------

    # >>> problem = glp_create_prob()
    # ... glp_read_lp(problem, None, "../tests/data/model.lp")
    # ... solution = solve_with_glpsol(problem)
    # ... print 'asdf'
    # 'asdf'
    # >>> print solution
    # 0.839784

    # Returns
    # -------
    # dict
    #     A dictionary containing the objective value (key ='objval')
    #     and variable primals.
    """
    from swiglpk import glp_get_row_name, glp_get_col_name, glp_write_lp, glp_get_num_rows, glp_get_num_cols

    row_ids = [glp_get_row_name(glp_prob, i) for i in range(1, glp_get_num_rows(glp_prob) + 1)]

    col_ids = [glp_get_col_name(glp_prob, i) for i in range(1, glp_get_num_cols(glp_prob) + 1)]

    tmp_file = tempfile.mktemp(suffix='.lp')
    # glp_write_mps(glp_prob, GLP_MPS_DECK, None, tmp_file)
    glp_write_lp(glp_prob, None, tmp_file)
    # with open(tmp_file, 'w') as tmp_handle:
    # tmp_handle.write(mps_string)
    cmd = ['glpsol', '--lp', tmp_file, '-w', tmp_file + '.sol', '--log', '/dev/null']
    term = check_output(cmd)
    log.info(term)

    with open(tmp_file + '.sol') as sol_handle:
        # print sol_handle.read()
        solution = dict()
        for i, line in enumerate(sol_handle.readlines()):
            if i <= 1 or line == '\n':
                pass
            elif i <= len(row_ids):
                solution[row_ids[i - 2]] = line.strip().split(' ')
            elif i <= len(row_ids) + len(col_ids) + 1:
                solution[col_ids[i - 2 - len(row_ids)]] = line.strip().split(' ')
            else:
                print(i)
                print(line)
                raise Exception("Argggh!")
    return solution


def glpk_read_cplex(path):
    """Reads cplex file and returns glpk problem.

    Returns
    -------
    glp_prob
        A glpk problems (same type as returned by glp_create_prob)
    """
    from swiglpk import glp_create_prob, glp_read_lp

    problem = glp_create_prob()
    glp_read_lp(problem, None, path)
    return problem


# noinspection PyBroadException
def list_available_solvers():
    """Determine available solver interfaces (with python bindings).

    Returns
    -------
    dict
        A dict like {'GLPK': True, 'GUROBI': False, ...}
    """
    solvers = dict(GUROBI=False, GLPK=False, MOSEK=False, CPLEX=False)
    try:
        import gurobipy

        solvers['GUROBI'] = True
        log.debug('Gurobi python bindings found at %s' % os.path.dirname(gurobipy.__file__))
    except:
        log.debug('Gurobi python bindings not available.')
    try:
        import swiglpk

        solvers['GLPK'] = True
        log.debug('GLPK python bindings found at %s' % os.path.dirname(swiglpk.__file__))
    except:
        log.debug('GLPK python bindings not available.')
    try:
        import mosek

        solvers['MOSEK'] = True
        log.debug('Mosek python bindings found at %s' % os.path.dirname(mosek.__file__))
    except:
        log.debug('Mosek python bindings not available.')
    try:
        import cplex

        solvers['CPLEX'] = True
        log.debug('CPLEX python bindings found at %s' % os.path.dirname(cplex.__file__))
    except:
        log.debug('CPLEX python bindings not available.')
    return solvers


if __name__ == '__main__':
    from swiglpk import glp_create_prob, glp_read_lp, glp_get_num_rows

    problem = glp_create_prob()
    glp_read_lp(problem, None, "../tests/data/model.lp")
    print("asdf", glp_get_num_rows(problem))
    solution = solve_with_glpsol(problem)
    print(solution['R_Biomass_Ecoli_core_w_GAM'])
        