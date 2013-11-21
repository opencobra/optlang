'''

@author: Nikolaus sonnenschein

   Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
   Technical University of Denmark.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
'''


import tempfile
# from .interface import Solver, Variable, Constraint
from interface import Variable, Solver
from glpk.glpkpi import *

class GlpkVariable(Variable):
    """..."""
    _type_to_glpk_kind = {'continuous': GLP_CV, 'integer': GLP_IV, 'binary': GLP_BV}

    def __init__(self, name, **kwargs):
        super(GlpkVariable, self).__init__(name, **kwargs)
        self.index = None

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):
            if name == 'name':
                super(GlpkVariable, self).__setattr__(name, value)
                glp_set_col_name(self.problem, self.index, self.name)
            
            elif name == 'index':
                super(GlpkVariable, self).__setattr__(name, value)
                _glpk_set_name()
            
            elif name == 'lb':
                super(GlpkVariable, self).__setattr__(name, value)
                self._glpk_set_lb()
            
            elif name == 'ub':
                super(GlpkVariable, self).__setattr__(name, value)
                _glpk_set_ub()
            
            elif name == 'obj':
                super(GlpkVariable, self).__setattr__(name, value)
                glp_set_obj_coef(self.problem, self.index, self.obj)
            
            elif name == 'type':
                super(GlpkVariable, self).__setattr__(name, value)
                try:
                    glpk_kind = self._type_to_glpk_kind[self.type]
                except KeyError, e:
                    raise Exception("GLPK cannot handle variables of type %s. \
                        The following variable types are available:\n" + " ".join(self._type_to_glpk_kind.keys()))
                glp_set_col_kind(self.problem, index, glpk_kind)
            elif hasattr(self, name):
                super(GlpkVariable, self).__setattr__(name, value)
            else:
                print "Funny smell"
        else:
            super(GlpkVariable, self).__setattr__(name, value)

    def _add_to_problem(self, problem):
        glp_add_cols(problem, 1)
        self.index = glp_get_num_cols(problem)
        self.problem = self.problem

    def _init(self):
        self._glpk_set_name()
        self._glpk_set_index()
        self._glpk_set_lb()
        self._glpk_set_ub()

    def _glpk_set_name(self):
        glp_set_col_name(self.problem, self.index, self.name)

    def _glpk_set_lb(self):
        if self.lb is None and self.ub is None:
            glp_set_col_bnds(self.problem, self.index, GLP_FR, 0., 0.) # 0.'s are ignored
        elif self.lb is None:
            glp_set_col_bnds(self.problem, self.index, GLP_UP, 0., self.ub) # 0. is ignored
        elif self.lb is self.ub:
            glp_set_col_bnds(self.problem, self.index, GLP_FX, self.lb, self.ub)
        elif self.lb != self.ub:
            glp_set_col_bnds(self.problem, self.index, GLP_DB, self.lb, self.ub)
        else:
            raise Exception("Something is wrong with the provided lower bound %f" % lb)

    def _glpk_set_ub(self):
        if self.lb is None and self.ub is None:
            glp_set_col_bnds(self.problem, index, GLP_FR, 0., 0.) # 0.'s are ignored
        elif self.lb is None:
            glp_set_col_bnds(self.problem, index, GLP_UP, 0., self.ub) # 0. is ignored
        elif self.lb is self.ub:
            glp_set_col_bnds(self.problem, index, GLP_FX, self.lb, self.ub)
        elif self.lb != self.ub:
            glp_set_col_bnds(self.problem, index, GLP_DB, self.lb, self.ub)
        else:
            raise Exception("Something is wrong with the provided lower bound %f" % lb)

    
# class GlpkConstraint(Constraint):
#     """GLPK solver interface"""
#     def __init__(self, **kwargs):
#         super(GlpkConstraint, self).__init__(**kwargs)

#     @property
#     def id(self):
#         return self.id
#     @id.setter
#     def id(self, value):
#         self.id = value
#         glp_set_row_name(self.problem, self.index, self.id)

#     @property
#     def rhs(self):
#         return self.rhs
#     @rhs.setter
#     def rhs(self, value):
#         self.rhs = value
#         glp_set_mat_row()
        

class GlpkSolver(Solver):
    """GLPK solver interface"""
    def __init__(self, problem=None):
        if problem == None:
            self.problem = glp_create_prob()
        elif isinstance(problem, glp_prob):
            self.problem = problem
        else:
            raise Exception, "Provided problem is not a valid GLPK model."
        
    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        glp_write_lp(self.problem, None, tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def _add_variable(self, variable):
        super(GlpkSolver, self)._add_variable(variable)
        variable._add_to_problem(self.problem)

    def _remove_variable(self, variable):
        raise NotImplementedError

    def _add_constraint(self, constraint):
        super(GlpkSolver, self)._add_constraint(constraint)
        ind, val = self._constraint_to_row(constraint)
        glp_add_rows(self.problem, 1)
        index = glp_get_num_rows(self.problem)
        glp_add_mat_row(self.problem, index, len(self.variables), ind, val)
        glp_set_col_bnds()

    def _constraint_to_row(self, constraint):
        ind = intArray(length + 1)
        val = doubleArray(length + 1)
        lhs = constraint.lhs
        rhs = constraint.rhs
        sense = type(constraint)
        return ind, val



if __name__ == '__main__':
    # x = Variable('x')
    # print x.problem
    # x.problem = 'test'
    # print x.problem

    x = GlpkVariable('x')
    x.name = "Test name"
    print x.name
    x.problem = "problem"
    print x.name
    x.name = "Test name2"

    # print dir(x)
    # print x.problem
    # x.problem = 'test'
    # print x.problem
            

#     # Example workflow
#     solver = GlpkSolver()
#     x = Variable('x', lb=0, ub=10)
#     y = Variable('y', lb=0, ub=10)
#     constr = Constraint(x + y > 3, name="constr1")
#     obj = Objective(2 x + y)
#     solver.add(x)
#     solver.add(y)
#     solver.add(constr)
#     solver.add(obj)
#     sol = solver.optimize()

    # from glpk.glpkpi import glp_read_lp
    # problem = glp_create_prob()
    # glp_read_lp(problem, None, "model.lp")
    # solver = GlpkSolver(problem=problem)
    # print solver
