# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import os
import tempfile
import glob
import tarfile
import sympy
import nose
from functools import partial
from glpk.glpkpi import *
from optlang.glpk_interface import Variable, Constraint, Model
from optlang.util import solve_with_glpsol


def read_netlib_sif_glpk(fhandle):
    tmp_file = tempfile.mktemp(suffix='.mps')
    with open(tmp_file, 'w') as tmp_handle:
        content = ''.join([s for s in fhandle if s.strip()])
        tmp_handle.write(content)
        fhandle.close()
    problem = glp_create_prob()
    glp_read_mps(problem, GLP_MPS_DECK, None, tmp_file)
    return problem

def check_dimensions(glpk_problem, model):
    """
    Tests that the glpk problem and the interface model have the same
    number of rows (constraints) and columns (variables).
    """
    assert glp_get_num_cols(glpk_problem) == len(model.variables)


def check_objval(glpk_problem, model):
    """
    Check that ...
    """
    smcp = glp_smcp()
    glp_simplex(glpk_problem, None)
    status = glp_get_status(glpk_problem)
    if status == GLP_OPT:
        glpk_problem_objval = glp_get_obj_val(glpk_problem)
    model.optimize()
    if model.status == 'optimal':
        model_objval = model.objective.value
    nose.tools.assert_almost_equal(glpk_problem_objval, model_objval)


def test_netlib(netlib_tar_path=os.path.join(os.path.dirname(__file__), 'data/netlib_lp_problems.tar.gz')):
    """
    Test netlib with glpk interface
    """
    tar = tarfile.open(netlib_tar_path)
    model_paths_in_tar = glob.fnmatch.filter(tar.getnames(), '*.SIF')

    for model_path_in_tar in model_paths_in_tar[0:20]:
        fhandle = tar.extractfile(model_path_in_tar)
        glpk_problem = read_netlib_sif_glpk(fhandle)
        model = Model(problem=glpk_problem)
        func = partial(check_dimensions, glpk_problem, model)
        func.description = "test_netlib_check_dimensions_%s (%s)" % (os.path.basename(model_path_in_tar), os.path.basename(str(__file__)))
        yield func

        func = partial(check_objval, glpk_problem, model)
        func.description = "test_netlib_check_objective_value_%s (%s)" % (os.path.basename(model_path_in_tar), os.path.basename(str(__file__)))
        yield func

if __name__ == '__main__':
    # tar = tarfile.open('data/netlib_lp_problems.tar.gz')
    # model_paths_in_tar = glob.fnmatch.filter(tar.getnames(), '*.SIF')
    # fhandle = tar.extractfile('./netlib/ADLITTLE.SIF')
    # glpk_problem = read_netlib_sif_glpk(fhandle)
    # glp_simplex(glpk_problem, None)
    # print glp_get_obj_val(glpk_problem)
    # print glpk_problem
    # fhandle = tar.extractfile('./netlib/ADLITTLE.SIF')
    # glpk_problem = read_netlib_sif_glpk(fhandle)
    # model = Model(problem=glpk_problem)
    # glp_simplex(glpk_problem, None)
    # model.optimize()
    # print model.objective.value
    # print model
    # test_netlib().next()
    nose.runmodule()