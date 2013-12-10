# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

"""Solver interface for the IBM ILOG CPLEX Optimization Studio solver.

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""

import logging
log = logging.getLogger(__name__)
import tempfile
import sympy
import cplex
import interface


class Variable(interface.Variable):
    """CPLEX variable interface."""
    # _type_to_glpk_kind = {'continuous': GLP_CV, 'integer': GLP_IV, 'binary': GLP_BV}

    def __init__(self, name, index=None, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)
        
    # @property
    # def index(self):
    #     try:
    #         i = glp_find_col(self.problem.problem, self.name)
    #         if i != 0:
    #             return i
    #         else:
    #             raise Exception("Could not determine row index for variable %s" % self)
    #     except:
    #         return None

    @property
    def primal(self):
        return dict(zip(self.problem.problem.variables.get_names(), self.problem.problem.solution.get_values()))[self.name]
    
    @property
    def dual(self):
        return dict(zip(self.problem.problem.variables.get_names(), self.problem.problem.solution.get_dual_values()))[self.name]

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):
            
            if name == 'lb':
                super(Variable, self).__setattr__(name, value)
                self.problem.problem.variables.set_lower_bounds(self.name, value)

            elif name == 'ub':
                super(Variable, self).__setattr__(name, value)
                self.problem.problem.variables.set_upper_bounds(self.name, value)
            
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
    """CPLEX constraint interface."""
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
    """CPLEX objective interface."""
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)


class Model(interface.Model):
    """CPLEX solver interface."""

    _cplex_status_to_status = {
        cplex.Cplex.solution.status.feasible: interface.FEASIBLE,
        cplex.Cplex.solution.status.infeasible: interface.INFEASIBLE,
        cplex.Cplex.solution.status.optimal: interface.OPTIMAL,
        cplex.Cplex.solution.status.unbounded: interface.UNBOUNDED,
        cplex.Cplex.solution.status.infeasible_or_unbounded: interface.INFEASIBLE_OR_UNBOUNDED
    }

    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        if problem is None:
            self.problem = cplex.Cplex()
        
        elif isinstance(problem, cplex.Cplex):
            self.problem = problem
            zipped_args = zip(self.problem.variables.get_names(),
                self.problem.variables.get_lower_bounds(),
                self.problem.variables.get_upper_bounds()
                )
            for name, lb, ub in zipped_args:
                var = Variable(name, lb=lb, ub=ub, problem=self)
                super(Model, self)._add_variable(var)  # This avoids adding the variable to the glpk problem
        else:
            raise Exception, "Provided problem is not a valid GLPK model."
    
    @property
    def objective(self):
        return self._objective
    @objective.setter
    def objective(self, value):
        self._objective = value

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        self.problem.write(tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def optimize(self):
        self.problem.solve()

        cplex_status = self.problem.solution.get_status()
        self.status = self._glpk_status_to_status[glpk_status]
        self.objective.value = blub.solution.get_objective_value()
        return self.status

    # def _add_variable(self, variable):
    #     super(Model, self)._add_variable(variable)
    #     glp_add_cols(self.problem, 1)
    #     index = glp_get_num_cols(self.problem)
    #     glp_set_col_name(self.problem, index, variable.name)
    #     variable.problem = self
    #     self._glpk_set_col_bounds(variable)
    #     return variable
    
    # def _remove_variable(self, variable):
    #     num = intArray(2)
    #     num[1] = variable.index
    #     glp_del_cols(self.problem, 1, num)
    #     super(Model, self)._remove_variable(variable)

    # def _add_constraint(self, constraint, sloppy=False):
    #     if sloppy is False:
    #         if not constraint.is_Linear:
    #             raise ValueError("GLPK only supports linear constraints. %s is not linear." % constraint)
    #     super(Model, self)._add_constraint(constraint, sloppy=sloppy)
    #     # for var in constraint.variables:
    #     #     if var.index is None:
    #     #         var.index = glp_find_col(self.problem, var.name)
    #         # if var.name not in self.variables:
    #         #     var
    #         #     self._add_variable(var)
    #     constraint.problem = self
    #     glp_add_rows(self.problem, 1)
    #     index = glp_get_num_rows(self.problem)
    #     glp_set_row_name(self.problem, index, constraint.name)
    #     # constraint.index = index
    #     num_vars = len(constraint.variables)
    #     index_array = intArray(num_vars + 1)
    #     value_array = doubleArray(num_vars + 1)
    #     if constraint.expression.is_Atom and constraint.expression.is_Symbol:
    #         var = constraint.expression
    #         index_array[1] = var.index
    #         value_array[1] = 1
    #     elif constraint.expression.is_Mul:
    #         args = constraint.expression.args
    #         if len(args) > 2:
    #             raise Exception("Term %s from constraint %s is not a proper linear term.", term, constraint)
    #         coeff = float(args[0])
    #         var = args[1]
    #         index_array[1] = var.index
    #         value_array[1] = coeff
    #     else:
    #         for i, term in enumerate(constraint.expression.args):
    #             args = term.args
    #             if args == ():
    #                 assert term.is_Symbol
    #                 coeff = 1
    #                 var = term
    #             elif len(args) == 2:
    #                 assert args[0].is_Number
    #                 assert args[1].is_Symbol
    #                 var = args[1]
    #                 coeff = float(args[0])
    #             elif leng(args) > 2:
    #                 raise Exception("Term %s from constraint %s is not a proper linear term.", term, constraint)
    #             index_array[i+1] = var.index
    #             value_array[i+1] = coeff
    #     glp_set_mat_row(self.problem, index, num_vars, index_array, value_array)
    #     self._glpk_set_row_bounds(constraint)

if __name__ == '__main__':

    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    z = Variable('z', lb=-100, ub=99.)
    constr = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')

    from cplex import Cplex
    problem = Cplex()
    problem.read("../tests/data/model.lp")
    

    # from optlang.util import solve_with_glpsol
    # print solve_with_glpsol(problem)
    
    solver = Model(problem=problem)
    # solver.add(z)
    # solver.add(constr)
    # # print solver
    # print solver.optimize()
    # print solver.objective