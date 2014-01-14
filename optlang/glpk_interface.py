# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

"""Interface for the GNU Linear Programming Kit (GLPK)

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""

import logging
log = logging.getLogger(__name__)
import tempfile
import sympy
from sympy.core.add import _unevaluated_Add
from sympy.core.mul import _unevaluated_Mul
from glpk.glpkpi import *
import interface


class Variable(interface.Variable):
    """..."""
    _type_to_glpk_kind = {'continuous': GLP_CV, 'integer': GLP_IV, 'binary': GLP_BV}

    def __init__(self, name, index=None, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)
        
    @property
    def index(self):
        try:
            i = glp_find_col(self.problem.problem, self.name)
            if i != 0:
                return i
            else:
                raise Exception("Could not determine row index for variable %s" % self)
        except:
            return None

    @property
    def primal(self):
        return glp_get_col_prim(self.problem.problem, self.index)
    
    @property
    def dual(self):
        return glp_get_col_dual(self.problem.problem, self.index)

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):
            
            if name == 'lb' or name == 'ub':
                super(Variable, self).__setattr__(name, value)
                self.problem._glpk_set_col_bounds(self)
            
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
    def __init__(self, expression, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)

    @property
    def index(self):
        try:
            i = glp_find_row(self.problem.problem, self.name)
            if i != 0:
                return i
            else:
                raise Exception("Could not determine row index for variable %s" % self)
        except:
            return None

    @property
    def primal(self):
        return glp_get_row_prim(self.problem.problem, self.index)
    
    @property
    def dual(self):
        return glp_get_row_dual(self.problem.problem, self.index)


    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):
        # if super(Constraint, self).__getattr__('problem'):
            
            if name == 'name':
                super(Constraint, self).__setattr__(name, value)
                self.problem._glpk_set_row_name(self)

            elif name == 'lb' or name == 'ub':
                super(Constraint, self).__setattr__(name, value)
                self.problem._glpk_set_row_bounds(self)

            else:
                super(Constraint, self).__setattr__(name, value)
        else:
            super(Constraint, self).__setattr__(name, value)

class Objective(interface.Objective):
    """docstring for Objective"""
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)

class Model(interface.Model):
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

        super(Model, self).__init__(*args, **kwargs)

        glp_term_out(GLP_OFF)
        self._smcp = glp_smcp()
        self._iocp = glp_iocp()
        glp_init_smcp(self._smcp)
        glp_init_iocp(self._iocp)
        self._smcp.msg_lev = GLP_MSG_ALL
        self._iocp.msg_lev = GLP_MSG_ON
        self._smcp.presolve = GLP_OFF
        self._iocp.presolve = GLP_OFF
        
        if problem is None:
            self.problem = glp_create_prob()
            glp_create_index(self.problem)
        
        elif isinstance(problem, glp_prob):
            self.problem = problem
            glp_create_index(self.problem)
            row_num = glp_get_num_rows(self.problem)
            col_num = glp_get_num_cols(self.problem)
            for i in xrange(1, col_num+1):
                var = Variable(
                            glp_get_col_name(self.problem, i),
                            lb=glp_get_col_lb(self.problem, i),
                            ub=glp_get_col_ub(self.problem, i),
                            problem=self
                        )
                super(Model, self)._add_variable(var)  # This avoids adding the variable to the glpk problem
            var = self.variables.values()

            for j in xrange(1, row_num+1):
                ia = intArray(col_num + 1)
                da = doubleArray(col_num + 1)
                nnz = glp_get_mat_row(self.problem, j, ia, da)
                lhs = _unevaluated_Add(*[da[i] * var[ia[i]-1] for i in range(1, nnz+1)])
                glpk_row_type = glp_get_row_type(self.problem, j)
                if glpk_row_type == GLP_FX:
                    row_lb = glp_get_row_lb(self.problem, j)
                    row_ub = row_lb
                elif glpk_row_type == GLP_LO:
                    row_lb = glp_get_row_lb(self.problem, j)
                    row_ub = None
                elif glpk_row_type == GLP_UP:
                    row_lb = None
                    row_ub = glp_get_row_ub(self.problem, j)
                elif glpk_row_type == GLP_DB:
                    row_lb = glp_get_row_lb(self.problem, j)
                    row_ub = glp_get_row_ub(self.problem, j)
                elif glpk_row_type == GLP_FR:
                    row_lb = None 
                    row_ub = None 
                else:
                    raise Exception("Currently, optlang does not support glpk row type " + str(glpk_row_type))
                    log.exception()
                if isinstance(lhs, int):
                    lhs = sympy.Integer(lhs)
                elif isinstance(lhs, float):
                    lhs = sympy.Real(lhs)
                super(Model, self)._add_constraint(
                        Constraint(lhs, lb=row_lb, ub=row_ub, name=glp_get_row_name(self.problem, j), problem=self),
                        sloppy=True
                    )
            
            term_generator = ((glp_get_obj_coef(self.problem, index), var[index-1]) for index in xrange(1, glp_get_num_cols(problem)+1))        
            # print _unevaluated_Add(*[_unevaluated_Mul(sympy.Real(term[0]), term[1]) for term in term_generator if term[0] != 0.])
            self._objective = interface.Objective(_unevaluated_Add(*[_unevaluated_Mul(sympy.Real(term[0]), term[1]) for term in term_generator if term[0] != 0.]), problem=self)
            self._objective.direction = {GLP_MIN:'min', GLP_MAX: 'max'}[glp_get_obj_dir(self.problem)]
        else:
            raise Exception, "Provided problem is not a valid GLPK model."
        glp_scale_prob(self.problem, GLP_SF_AUTO)
    
    @property
    def objective(self):
        return self._objective
    @objective.setter
    def objective(self, value):
        self._objective = value
        for i in xrange(1, glp_get_num_cols(self.problem)+1):
            glp_set_obj_coef(self.problem, i, 0.)
        expression = self._objective.expression
        if expression.is_Atom:
            glp_set_obj_coef(self.problem, expression.index, 1.)
        if expression.is_Mul:
            coeff, var = expression.args
            glp_set_obj_coef(self.problem, var.index, float(coeff))
        elif expression.is_Add:
            for term in expression.args:
                coeff, var = term.args
                glp_set_obj_coef(self.problem, var.index, float(coeff))
        else:
            raise ValueError("Provided objective %s doesn't seem to be appropriate." % self._objective)
        glp_set_obj_dir(self.problem, {'min': GLP_MIN, 'max': GLP_MAX}[self._objective.direction])

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        glp_write_lp(self.problem, None, tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def optimize(self):
        glp_simplex(self.problem, self._smcp)
        glpk_status = glp_get_status(self.problem)
        self.status = self._glpk_status_to_status[glpk_status]
        self.objective.value = glp_get_obj_val(self.problem)
        # for var_id, variable in self.variables.items():
        #     variable.primal = glp_get_col_prim(self.problem, variable.index)
        #     variable.dual = glp_get_col_dual(self.problem, variable.index)
        # for constr_id, constraint in self.constraints.items():
        #     constraint.primal = glp_get_row_prim(self.problem, constraint.index)
        #     constraint.dual = glp_get_row_dual(self.problem, constraint.index)
        return self.status

    def _add_variable(self, variable):
        super(Model, self)._add_variable(variable)
        glp_add_cols(self.problem, 1)
        index = glp_get_num_cols(self.problem)
        glp_set_col_name(self.problem, index, variable.name)
        variable.problem = self
        self._glpk_set_col_bounds(variable)
        return variable
    
    def _remove_variable(self, variable):
        num = intArray(2)
        num[1] = variable.index
        glp_del_cols(self.problem, 1, num)
        super(Model, self)._remove_variable(variable)

    def _add_constraint(self, constraint, sloppy=False):
        if sloppy is False:
            if not constraint.is_Linear:
                raise ValueError("GLPK only supports linear constraints. %s is not linear." % constraint)
        super(Model, self)._add_constraint(constraint, sloppy=sloppy)
        # for var in constraint.variables:
        #     if var.index is None:
        #         var.index = glp_find_col(self.problem, var.name)
            # if var.name not in self.variables:
            #     var
            #     self._add_variable(var)
        constraint.problem = self
        glp_add_rows(self.problem, 1)
        index = glp_get_num_rows(self.problem)
        glp_set_row_name(self.problem, index, constraint.name)
        # constraint.index = index
        num_vars = len(constraint.variables)
        index_array = intArray(num_vars + 1)
        value_array = doubleArray(num_vars + 1)
        if constraint.expression.is_Atom and constraint.expression.is_Symbol:
            var = constraint.expression
            index_array[1] = var.index
            value_array[1] = 1
        elif constraint.expression.is_Mul:
            args = constraint.expression.args
            if len(args) > 2:
                raise Exception("Term %s from constraint %s is not a proper linear term.", term, constraint)
            coeff = float(args[0])
            var = args[1]
            index_array[1] = var.index
            value_array[1] = coeff
        else:
            for i, term in enumerate(constraint.expression.args):
                args = term.args
                if args == ():
                    assert term.is_Symbol
                    coeff = 1
                    var = term
                elif len(args) == 2:
                    assert args[0].is_Number
                    assert args[1].is_Symbol
                    var = args[1]
                    coeff = float(args[0])
                elif leng(args) > 2:
                    raise Exception("Term %s from constraint %s is not a proper linear term.", term, constraint)
                index_array[i+1] = var.index
                value_array[i+1] = coeff
        glp_set_mat_row(self.problem, index, num_vars, index_array, value_array)
        self._glpk_set_row_bounds(constraint)

    # def _glpk_set_col_name(self, variable):
    #     glp_set_col_name(self.problem, variable.index, variable.name)

    # def _glpk_set_row_name(self, constraint):
    #     glp_set_row_name(self.problem, constraint.index, constraint.name)

    def _glpk_set_col_bounds(self, variable):
        if variable.lb is None and variable.ub is None:
            glp_set_col_bnds(self.problem, variable.index, GLP_FR, 0., 0.) # 0.'s are ignored
        elif variable.lb is None:
            glp_set_col_bnds(self.problem, variable.index, GLP_UP, 0., float(variable.ub)) # 0. is ignored
        elif variable.ub is None:
            glp_set_col_bnds(self.problem, variable.index, GLP_LO, float(variable.lb), 0.) # 0. is ignored
        elif variable.lb == variable.ub:
            glp_set_col_bnds(self.problem, variable.index, GLP_FX, float(variable.lb), float(variable.lb))
        elif variable.lb < variable.ub:
            glp_set_col_bnds(self.problem, variable.index, GLP_DB, float(variable.lb), float(variable.ub))
        elif variable.lb > variable.ub:
            raise ValueError("Lower bound %f is larger than upper bound %f in variable %s" %
                (variable.lb, variable.ub, variable))
        else:
            raise Exception("Something is wrong with the provided bounds %f and %f in variable %s" %
                (variable.lb, variable.ub, variable))

    def _glpk_set_row_bounds(self, constraint):
        if constraint.lb is None and constraint.ub is None:
            glp_set_row_bnds(self.problem, constraint.index, GLP_FR, 0., 0.) # 0.'s are ignored
        elif constraint.lb is None:
            glp_set_row_bnds(self.problem, constraint.index, GLP_UP, 0., float(constraint.ub)) # 0. is ignored
        elif constraint.ub is None:
            glp_set_row_bnds(self.problem, constraint.index, GLP_LO, float(constraint.lb), 0.) # 0. is ignored
        elif constraint.lb is constraint.ub:
            glp_set_row_bnds(self.problem, constraint.index, GLP_FX, float(constraint.lb), float(constraint.lb))
        elif constraint.lb < constraint.ub:
            print 'Double bounded constraint!'
            print constraint.lb, constraint.ub, constraint.lb - constraint.ub
            glp_set_row_bnds(self.problem, constraint.index, GLP_DB, float(constraint.lb), float(constraint.ub))
        elif constraint.lb > constraint.ub:
            raise ValueError("Lower bound %f is larger than upper bound %f in constraint %s" %
                (constraint.lb, constraint.ub, constraint))
        else:
            raise Exception("Something is wrong with the provided bounds %f and %f in constraint %s" %
                (constraint.lb, constraint.ub, constraint))


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    z = Variable('z', lb=-100, ub=99.)
    constr = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')
    print 1/0
    from glpk.glpkpi import glp_read_lp
    problem = glp_create_prob()
    # print glp_get_obj_coef(problem, 0)
    glp_read_lp(problem, None, "../tests/data/model.lp")

    # from optlang.util import solve_with_glpsol
    # print solve_with_glpsol(problem)
    
    solver = Model(problem=problem)
    solver.add(z)
    solver.add(constr)
    # print solver
    print solver.optimize()
    print solver.objective