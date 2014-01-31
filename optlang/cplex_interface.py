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
from sympy.core.add import _unevaluated_Add
from sympy.core.mul import _unevaluated_Mul
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

    _vtype_to_cplex_type = {'continuous': cplex.Cplex.variables.type.continuous,
        'integer': cplex.Cplex.variables.type.continuous,
        'binary': cplex.Cplex.variables.type.continuous
        }

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
            zipped_var_args = zip(self.problem.variables.get_names(),
                self.problem.variables.get_lower_bounds(),
                self.problem.variables.get_upper_bounds()
                )
            for name, lb, ub in zipped_var_args:
                var = Variable(name, lb=lb, ub=ub, problem=self)
                super(Model, self)._add_variable(var)  # This avoids adding the variable to the glpk problem
            zipped_constr_args = zip(self.problem.linear_constraints.get_names(),
                self.problem.linear_constraints.get_rows(),
                self.problem.linear_constraints.get_senses(),
                self.problem.linear_constraints.get_rhs()
                )
            var = self.variables.values()
            for name, row, sense, rhs in zipped_constr_args:
                lhs = _unevaluated_Add(*[val * var[i-1] for i, val in zip(row.ind, row.val)])
                if isinstance(lhs, int):
                    lhs = sympy.Integer(lhs)
                elif isinstance(lhs, float):
                    lhs = sympy.Real(lhs)
                if sense == 'E':
                    constr = Constraint(lhs, lb=rhs, ub=rhs, name=name, problem=self)
                elif sense == 'G':
                    constr = Constraint(lhs, lb=rhs, name=name, problem=self)
                elif sense == 'L':
                    constr = Constraint(lhs, ub=rhs, name=name, problem=self)
                elif sense == 'R':
                    raise Exception, 'optlang does not provide support for CPLEX range constraints yet.'
                else:
                    raise Exception, '%s is not a recognized constraint sense.' % sense
                super(Model, self)._add_constraint(
                        constr,
                        sloppy=True
                    )
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

    def _cplex_sense_to_sympy(self, sense, translation={'E': '==', 'L': '<', 'G': '>'}):
        try:
            return translation[sense]
        except KeyError, e:
            print ' '.join('Sense', sense, 'is not a proper relational operator, e.g. >, <, == etc.')
            print e

    def _add_variable(self, variable):
        super(Model, self)._add_variable(variable)
        if variable.lb == None:
            lb = -cplex.infinity
        else:
            lb = variable.lb
        if variable.ub == None:
            ub = cplex.infinity
        else:
            ub = variable.ub
        vtype = self._vtype_to_cplex_type[variable.type]
        self.problem.variables.add([0.], lb=[lb], ub=[ub], types=[vtype], names=[variable.name])
        variable.problem = self
        return variable
    
    # def _remove_variable(self, variable):
    #     num = intArray(2)
    #     num[1] = variable.index
    #     glp_del_cols(self.problem, 1, num)
    #     super(Model, self)._remove_variable(variable)

    def _add_constraint(self, constraint, sloppy=False):
        if sloppy is False:
            if not constraint.is_Linear and not constraint.is_Quadratic:
                raise ValueError("CPLEX only supports linear or quadratic constraints. %s is neither linear nor quadratic." % constraint)
        super(Model, self)._add_constraint(constraint, sloppy=sloppy)
        for var in constraint.variables:
            if var.name not in self.variables:
                self._add_variable(var)
        if constraint.is_Linear:
            if constraint.expression.is_Atom and constraint.expression.is_Symbol:
                ind = constraint.expression.name
                val = 1.
            elif constraint.expression.is_Mul:
                args = constraint.expression.args
                if len(args) > 2:
                    raise Exception("Term %s from constraint %s is not a proper linear term.", term, constraint)
                val = float(args[0])
                ind = args[1].name
            else:
                for i, term in enumerate(constraint.expression.args):
                    args = term.args
                    if args == ():
                        assert term.is_Symbol
                        val = 1.
                        ind = term.name
                    elif len(args) == 2:
                        assert args[0].is_Number
                        assert args[1].is_Symbol
                        ind = args[1].name
                        val = float(args[0])
                    elif leng(args) > 2:
                        raise Exception("Term %s from constraint %s is not a proper linear term.", term, constraint)
                    
            self.problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[ind], val=[val])], senses='', rhs=[0], range_values=[], names=[constraint.name])
        constraint.problem = self
        return constraint

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