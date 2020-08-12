# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import glob
import logging
import os
import pickle
import tarfile
import tempfile
import unittest
from functools import partial

import nose
import six

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if six.PY3:
    nose.SkipTest('Skipping because py3')
else:
    try:
        import gurobipy

        from optlang.gurobi_interface import Model

        with open(os.path.join(os.path.dirname(__file__), 'data/the_final_netlib_results.pcl'), 'rb') as fhandle:
            THE_FINAL_NETLIB_RESULTS = pickle.load(fhandle)


        class TestClass(object):
            """docstring for TestClass"""

            def __init__(self, arg):
                super(TestClass, self).__init__()
                self.arg = arg


        # noinspection PyShadowingNames
        def read_netlib_sif_gurobi(fhandle):
            tmp_file = tempfile.mktemp(suffix='.mps')
            with open(tmp_file, 'w') as tmp_handle:
                content = ''.join([str(s) for s in fhandle if str(s).strip()])
                tmp_handle.write(content)
                fhandle.close()
                log.debug(tmp_file)
            problem = gurobipy.read(tmp_file)
            return problem


        def check_dimensions(problem, model):
            """
            Tests that the glpk problem and the interface model have the same
            number of rows (constraints) and columns (variables).
            """
            assert len(problem.getVars()) == len(model.variables)


        def check_objval(problem, model_objval):
            """
            Check that ...
            """
            if problem.getAttr('Status') == gurobipy.GRB.OPTIMAL:
                problem_objval = problem.getAttr('ObjVal')
            else:
                problem_objval = None
            nose.tools.assert_almost_equal(problem_objval, model_objval, places=4)


        def check_objval_against_the_final_netlib_results(netlib_id, model_objval):
            relative_error = abs(1 - (model_objval / float(THE_FINAL_NETLIB_RESULTS[netlib_id]['Objvalue'])))
            print(relative_error)
            nose.tools.assert_true(relative_error < 0.01)
            # nose.tools.assert_almost_equal(model_objval, float(THE_FINAL_NETLIB_RESULTS[netlib_id]['Objvalue']), places=4)


        def test_netlib(netlib_tar_path=os.path.join(os.path.dirname(__file__), 'data/netlib_lp_problems.tar.gz')):
            """
            Test netlib with glpk interface
            """
            tar = tarfile.open(netlib_tar_path)
            model_paths_in_tar = glob.fnmatch.filter(tar.getnames(), '*.SIF')

            for model_path_in_tar in model_paths_in_tar:
                print(model_path_in_tar)
                netlib_id = os.path.basename(model_path_in_tar).replace('.SIF', '')
                # TODO: get the following problems to work
                # E226 seems to be a MPS related problem, see http://lists.gnu.org/archive/html/bug-glpk/2003-01/msg00003.html
                if netlib_id in ('AGG', 'E226', 'SCSD6', 'BLEND', 'DFL001', 'FORPLAN', 'GFRD-PNC', 'SIERRA'):
                    # def test_skip(netlib_id):
                    # raise SkipTest('Skipping netlib problem %s ...' % netlib_id)
                    # test_skip(netlib_id)
                    # class TestWeirdNetlibProblems(unittest.TestCase):

                    # @unittest.skip('Skipping netlib problem')
                    # def test_fail():
                    # pass
                    continue
                # TODO: For now, test only models that are covered by the final netlib results
                else:
                    if netlib_id not in THE_FINAL_NETLIB_RESULTS.keys():
                        continue
                    fhandle = tar.extractfile(model_path_in_tar)
                    problem = read_netlib_sif_gurobi(fhandle)
                    model = Model(problem=problem)
                    model.configuration.presolve = True
                    model.configuration.verbosity = 3
                    func = partial(check_dimensions, problem, model)
                    func.description = "test_netlib_check_dimensions_%s (%s)" % (
                        netlib_id, os.path.basename(str(__file__)))
                    yield func

                    model.optimize()
                    if model.status == 'optimal':
                        model_objval = model.objective.value
                    else:
                        raise Exception('No optimal solution found for netlib model %s' % netlib_id)

                    func = partial(check_objval, problem, model_objval)
                    func.description = "test_netlib_check_objective_value_%s (%s)" % (
                        netlib_id, os.path.basename(str(__file__)))
                    yield func

                    func = partial(check_objval_against_the_final_netlib_results, netlib_id, model_objval)
                    func.description = "test_netlib_check_objective_value__against_the_final_netlib_results_%s (%s)" % (
                        netlib_id, os.path.basename(str(__file__)))
                    yield func

                    if not os.getenv('TRAVIS', False):
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

    except ImportError as e:

        if str(e).find('gurobipy') >= 0:
            class TestMissingDependency(unittest.TestCase):

                @unittest.skip('Missing dependency - ' + str(e))
                def test_fail(self):
                    pass
        else:
            raise

if __name__ == '__main__':
    # tar = tarfile.open('data/netlib_lp_problems.tar.gz')
    # model_paths_in_tar = glob.fnmatch.filter(tar.getnames(), '*.SIF')
    # for model_path_in_tar in model_paths_in_tar:
    # try:
    #
    # fhandle = tar.extractfile(model_path_in_tar)
    # problem = read_netlib_sif_cplex(fhandle)
    #
    # except cplex.exceptions.CplexSolverError, e:
    # print model_path_in_tar
    # print problem
    # print problem.get_problem_name()
    # print e
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
