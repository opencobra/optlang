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

import logging
log = logging.getLogger(__name__)
import tempfile
import sympy
import interface
from glpk.glpkpi import *

class Variable(interface.Variable):
    """..."""
    _type_to_glpk_kind = {'continuous': GLP_CV, 'integer': GLP_IV, 'binary': GLP_BV}

    def __init__(self, name, index=None, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)
        self.index = index

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):
            if name == 'name':
                super(Variable, self).__setattr__(name, value)
                self.problem._glpk_set_col_name(self)
            
            elif name == 'lb':
                super(Variable, self).__setattr__(name, value)
                self.problem._glpk_set_col_lb(self)
            
            elif name == 'ub':
                super(Variable, self).__setattr__(name, value)
                self.problem._glpk_set_col_ub(self)
            
            # elif name == 'obj':
            #     super(Variable, self).__setattr__(name, value)
            #     glp_set_obj_coef(self.problem, self.index, self.obj)
            
            # elif name == 'type':
            #     super(Variable, self).__setattr__(name, value)
            #     try:
            #         glpk_kind = self._type_to_glpk_kind[self.type]
            #     except KeyError, e:
            #         raise Exception("GLPK cannot handle variables of type %s. \
            #             The following variable types are available:\n" + " ".join(self._type_to_glpk_kind.keys()))
            #     glp_set_col_kind(self.problem, index, glpk_kind)
            # elif hasattr(self, name):
            #     super(Variable, self).__setattr__(name, value)
            else:
                super(Variable, self).__setattr__(name, value)
        else:
            super(Variable, self).__setattr__(name, value)



 
class Constraint(interface.Constraint):
    """GLPK solver interface"""
    def __init__(self, expression, index=None, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)
        self.index = index

class Solver(interface.Solver):
    """GLPK solver interface"""

    _glpk_status_to_status = {
        GLP_UNDEF:interface.UNDEFINED,
        GLP_FEAS:interface.FEASIBLE,
        GLP_INFEAS:interface.INFEASIBLE,
        GLP_NOFEAS:interface.INFEASIBLE,
        GLP_OPT:interface.OPTIMAL,
        GLP_UNBND:interface.UNBOUNDED
    }

    def __init__(self, problem=None, *args, **kwargs):

        super(Solver, self).__init__(*args, **kwargs)

        glp_term_out(GLP_OFF)
        self._smcp = glp_smcp()
        self._iocp = glp_iocp()
        glp_init_smcp(self._smcp)
        glp_init_iocp(self._iocp)
        self._smcp.msg_lev = GLP_MSG_ON
        self._iocp.msg_lev = GLP_MSG_ON
        self._smcp.presolve = GLP_ON
        self._iocp.presolve = GLP_ON
        
        if problem is None:
            self.problem = glp_create_prob()
        
        elif isinstance(problem, glp_prob):
            self.problem = problem
            tmp_file = tempfile.mktemp(suffix=".lp")
            
            glp_write_lp(self.problem, None, tmp_file)
            cplex_form = open(tmp_file).read()
            
            row_num = glp_get_num_rows(self.problem)
            col_num = glp_get_num_cols(self.problem)
            for i in xrange(1, col_num+1):
                var = Variable(
                            glp_get_col_name(self.problem, i),
                            lb=glp_get_col_lb(self.problem, i),
                            ub=glp_get_col_ub(self.problem, i),
                            problem=self,
                            index=i
                        )
                super(Solver, self)._add_variable(var)  # This avoids adding the variable to the glpk problem
            for j in xrange(1, row_num+1):
                ia = intArray(col_num + 1)
                da = doubleArray(col_num + 1)
                nnz = glp_get_mat_row(self.problem, j, ia, da)
                log.debug("nnz: %d", nnz)
                # for i in range(1, nnz+1):
                #     print "ai", ia[i]
                #     print "da", da[i]
                #     print len(self.variables.values())
                #     print self.variables.values()
                lhs = sum([da[i] * self.variables.values()[ia[i]-1] for i in range(1, nnz+1)])
                # print lhs
                glpk_row_type = glp_get_row_type(self.problem, j)
                if glpk_row_type == GLP_FX:
                    sense = '=='
                    rhs = glp_get_row_lb(self.problem, j)
                elif glpk_row_type == GLP_LO:
                    sense = '>'
                    rhs = glp_get_row_lb(self.problem, j)
                elif glpk_row_type == GLP_UP:
                    sense = '<'
                    rhs = glp_get_row_ub(self.problem, j)
                else:
                    raise Exception("Currently, optlang does not support glpk row type " + str(glpk_row_type))
                    log.exception()
                super(Solver, self)._add_constraint(
                        Constraint(
                            sympy.Rel(
                                lhs,
                                rhs,
                                sense
                                ),
                            name=glp_get_row_name(self.problem, j),
                            problem=self,
                            index=j
                            )
                    )
        else:
            raise Exception, "Provided problem is not a valid GLPK model."

        glp_scale_prob(self.problem, GLP_SF_AUTO)
        
    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        glp_write_lp(self.problem, None, tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def optimize(self):
        solution = dict()
        solution['variables'] = dict()
        solution['constraints'] = dict()
        glp_simplex(self.problem, self._smcp)
        glpk_status = glp_get_status(self.problem)
        self.status = self._glpk_status_to_status[glpk_status]
        solution['status'] = self.status
        for var in self.variables.values():
            var.primal = glp_get_col_prim(self.problem, var.index)
            var.dual = glp_get_col_dual(self.problem, var.index)
            solution['variables'][var.name]  = (var.primal, var.dual)
        for constraint in self.constraints.values():
            constraint.primal = glp_get_row_prim(self.problem, constraint.index)
            constraint.dual = glp_get_row_dual(self.problem, constraint.index)
            solution['constraints'][constraint.name]  = (constraint.primal, constraint.dual)
        return solution


    def _add_variable(self, variable):
        super(Solver, self)._add_variable(variable)
        glp_add_cols(problem, 1)
        variable.index = glp_get_num_cols(problem)
        variable.problem = self.problem
        self._glpk_set_col_name(variable)
        # self._glpk_set_col_lb(variable)
        # self._glpk_set_col_ub(variable)
        self._glpk_set_col_bounds(variable)
    
    def _remove_variable(self, variable):
        raise NotImplementedError

    def _add_constraint(self, constraint):   
        super(Solver, self)._add_constraint(constraint)
        for var in constraint.variables:
            if var.name not in self.variables:
                self._add_variable(var)
        glp_add_rows(self.problem, 1)
        index = glp_get_num_rows(self.problem)
        constraint.index = index
        constraint.problem = self
        num_vars = len(constraint.variables)
        index_array = intArray(num_vars + 1)
        value_array = doubleArray(num_vars + 1)
        for i, term in enumerate(constraint.expression.args):
            args = term.args
            if len(args) > 2 or not args[0].is_Number or not args[1].is_Symbol:
                raise Exception("GLPK only supports linear constraints. %s is not linear." % constraint)
            var = args[1]
            coeff = float(args[0])
            index_array[i+1] = var.index
            value_array[i+1] = coeff
        glp_set_row_name(self.problem, index, constraint.name)
        glp_set_mat_row(self.problem, index, num_vars, index_array, value_array)
        # sense = constraint.expression.rel_op;
        # glp_set_row_bnds(self.problem, index, GLP_FX, constraint.expression.rhs, 10.)
        self._glpk_set_row_bounds(constraint)

    def _glpk_set_col_name(self, variable):
        glp_set_col_name(self.problem, variable.index, variable.name)

    def _glpk_set_col_bounds(self, variable):
        if variable.lb is None and variable.ub is None:
            glp_set_col_bnds(self.problem, variable.index, GLP_FR, 0., 0.) # 0.'s are ignored
        elif variable.lb is None:
            glp_set_col_bnds(self.problem, variable.index, GLP_UP, 0., variable.ub) # 0. is ignored
        elif variable.lb is variable.ub:
            glp_set_col_bnds(self.problem, variable.index, GLP_FX, variable.lb, variable.lb)
        elif variable.lb < variable.ub:
            glp_set_col_bnds(self.problem, variable.index, GLP_DB, variable.lb, variable.ub)
        elif variable.lb > variable.ub:
            raise Exception("Lower bound %f is larger thane upper bound %f in variable %s" %
                (variable.lb, variable.ub, variable))
        else:
            raise Exception("Something is wrong with the provided bounds %f and %f in variable %s" %
                (variable.lb, variable.ub, variable))

    def _glpk_set_row_bounds(self, constraint):
        if constraint.lb is None and constraint.ub is None:
            glp_set_row_bnds(self.problem, constraint.index, GLP_FR, 0., 0.) # 0.'s are ignored
        elif constraint.lb is None:
            glp_set_row_bnds(self.problem, constraint.index, GLP_UP, 0., constraint.ub) # 0. is ignored
        elif constraint.lb is constraint.ub:
            glp_set_row_bnds(self.problem, constraint.index, GLP_FX, constraint.lb, constraint.lb)
        elif constraint.lb < constraint.ub:
            glp_set_row_bnds(self.problem, constraint.index, GLP_DB, constraint.lb, constraint.ub)
        elif constraint.lb > constraint.ub:
            raise Exception("Lower bound %f is larger thane upper bound %f in constraint %s" %
                (constraint.lb, constraint.ub, constraint))
        else:
            raise Exception("Something is wrong with the provided bounds %f and %f in constraint %s" %
                (constraint.lb, constraint.ub, constraint))


if __name__ == '__main__':

    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    z = Variable('z', lb=-100, ub=99.)
    constr = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')

    from glpk.glpkpi import glp_read_lp
    problem = glp_create_prob()
    glp_read_lp(problem, None, "tests/data/model.lp")
    
    solver = Solver(problem=problem)
    solver.add(z)
    solver.add(constr)
    print solver
    print solver.optimize()