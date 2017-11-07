# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import os
import random

import nose
from optlang import glpk_exact_interface
from optlang.tests import test_glpk_interface
from optlang.glpk_exact_interface import Variable, Constraint, Model, Objective

random.seed(666)
TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')

class VariableTestCase(test_glpk_interface.VariableTestCase):
    interface = glpk_exact_interface


class ConstraintTestCase(test_glpk_interface.ConstraintTestCase):
    interface = glpk_exact_interface


class ObjectiveTestCase(test_glpk_interface.ObjectiveTestCase):
    interface = glpk_exact_interface


class ConfigurationTestCase(test_glpk_interface.ConfigurationTestCase):
    interface = glpk_exact_interface


class ModelTestCase(test_glpk_interface.ModelTestCase):
    interface = glpk_exact_interface


if __name__ == '__main__':
    nose.runmodule()
