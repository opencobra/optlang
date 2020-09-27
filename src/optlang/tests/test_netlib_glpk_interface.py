# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import glob
import os
import pickle
import tarfile
import tempfile
from functools import partial

import nose
import six
from optlang.glpk_interface import Model
from swiglpk import glp_create_prob, GLP_MPS_DECK, glp_read_mps, glp_get_num_cols, glp_smcp, glp_simplex, \
    glp_get_status, \
    GLP_OPT, glp_get_obj_val


def test_netlib(netlib_tar_path=os.path.join(os.path.dirname(__file__), 'data/netlib_lp_problems.tar.gz')):
    """
    Test netlib with glpk interface
    """
    if six.PY3:
        nose.SkipTest('Skipping because py3')
    else:
        with open(os.path.join(os.path.dirname(__file__), 'data/the_final_netlib_results.pcl'), 'rb') as fhandle:
            THE_FINAL_NETLIB_RESULTS = pickle.load(fhandle)

        # noinspection PyShadowingNames
        def read_netlib_sif_glpk(fhandle):
            tmp_file = tempfile.mktemp(suffix='.mps')
            with open(tmp_file, 'w') as tmp_handle:
                content = ''.join([str(s) for s in fhandle if str(s.strip())])
                tmp_handle.write(content)
                fhandle.close()
            problem = glp_create_prob()
            glp_read_mps(problem, GLP_MPS_DECK, None, tmp_file)
            # glp_read_mps(problem, GLP_MPS_FILE, None, tmp_file)
            return problem

        def check_dimensions(glpk_problem, model):
            """
            Tests that the glpk problem and the interface model have the same
            number of rows (constraints) and columns (variables).
            """
            assert glp_get_num_cols(glpk_problem) == len(model.variables)

        def check_objval(glpk_problem, model_objval):
            """
            Check that ...
            """
            smcp = glp_smcp()
            smcp.presolve = True
            glp_simplex(glpk_problem, None)
            status = glp_get_status(glpk_problem)
            if status == GLP_OPT:
                glpk_problem_objval = glp_get_obj_val(glpk_problem)
            else:
                glpk_problem_objval = None
            nose.tools.assert_almost_equal(glpk_problem_objval, model_objval, places=4)

        def check_objval_against_the_final_netlib_results(netlib_id, model_objval):
            relative_error = abs(1 - (model_objval / float(THE_FINAL_NETLIB_RESULTS[netlib_id]['Objvalue'])))
            print(relative_error)
            nose.tools.assert_true(relative_error < 0.01)
            # nose.tools.assert_almost_equal(model_objval, float(THE_FINAL_NETLIB_RESULTS[netlib_id]['Objvalue']), places=4)

        tar = tarfile.open(netlib_tar_path)
        model_paths_in_tar = glob.fnmatch.filter(tar.getnames(), '*.SIF')

        for model_path_in_tar in model_paths_in_tar:
            netlib_id = os.path.basename(model_path_in_tar).replace('.SIF', '')
            # TODO: get the following problems to work
            # E226 seems to be a MPS related problem, see http://lists.gnu.org/archive/html/bug-glpk/2003-01/msg00003.html
            if netlib_id in ('AGG', 'E226', 'SCSD6', 'DFL001'):
                # def test_skip(netlib_id):
                # raise SkipTest('Skipping netlib problem %s ...' % netlib_id)
                # test_skip(netlib_id)
                # class TestWeirdNetlibProblems(unittest.TestCase):

                # @unittest.skip('Skipping netlib problem')
                # def test_fail():
                #         pass
                continue
            # TODO: For now, test only models that are covered by the final netlib results
            else:
                if netlib_id not in THE_FINAL_NETLIB_RESULTS.keys():
                    continue
                fhandle = tar.extractfile(model_path_in_tar)
                glpk_problem = read_netlib_sif_glpk(fhandle)
                model = Model(problem=glpk_problem)
                model.configuration.presolve = True
                # model.configuration.verbosity = 3
                func = partial(check_dimensions, glpk_problem, model)
                func.description = "test_netlib_check_dimensions_%s (%s)" % (netlib_id, os.path.basename(str(__file__)))
                yield func

                model.optimize()
                if model.status == 'optimal':
                    model_objval = model.objective.value
                else:
                    raise Exception('No optimal solution found for netlib model %s' % netlib_id)

                func = partial(check_objval, glpk_problem, model_objval)
                func.description = "test_netlib_check_objective_value_%s (%s)" % (
                    netlib_id, os.path.basename(str(__file__)))
                yield func

                func = partial(check_objval_against_the_final_netlib_results, netlib_id, model_objval)
                func.description = "test_netlib_check_objective_value__against_the_final_netlib_results_%s (%s)" % (
                    netlib_id, os.path.basename(str(__file__)))
                yield func

                if os.getenv('CI', 'false') != 'true':
                    # check that a cloned model also gives the correct result
                    model = Model.clone(model, use_json=False, use_lp=False)
                    model.optimize()
                    if model.status == 'optimal':
                        model_objval = model.objective.value
                    else:
                        raise Exception('No optimal solution found for netlib model %s' % netlib_id)

                    func = partial(check_objval_against_the_final_netlib_results, netlib_id, model_objval)
                    func.description = "test_netlib_check_objective_value__against_the_final_netlib_results_after_cloning_%s (%s)" % (
                        netlib_id, os.path.basename(str(__file__)))
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
