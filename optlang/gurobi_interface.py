# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from warnings import warn
warn("Be careful! The GUROBI interface is still under construction ...")

import logging
log = logging.getLogger(__name__)
import tempfile
import interface
from interface import Constraint, Objective
import sympy
import gurobipy


class Variable(interface.Variable):
    
    def __init__(self, *args, **kwargs):
        super(Variable, self).__init__(*args, **kwargs)
        self._internal_var = None

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):

            # if name == 'problem':
            #     raise AttributeError("problem is read only")
            
            if name == 'lb':
                super(Variable, self).__setattr__(name, value)
                self._internal_var.setAttr("lb", value)

            elif name == 'ub':
                super(Variable, self).__setattr__(name, value)
                self._internal_var.setAttr("ub", value)

            elif name == 'type':
                super(Variable, self).__setattr__(name, value)
                self._internal_var.setAttr("vtype", value)
        
        elif getattr(self, '_internal_var', None):

            if name == 'primal':
                super(Variable, self).__setattr__(name, value)
                return self._internal_var.setAttr('X', value)
            
            elif name == 'dual':
                super(Variable, self).__setattr__(name, value)
                return self._internal_var.setAttr('RC', value)
        
        else:
            
            super(Variable, self).__setattr__(name, value)

    def __getattr__(self, name):
        
        if hasattr(self, '_internal_var'):
            if name == 'primal':
                primal = self._internal_var.getAttr('X')
                # super(Variable, self).__setattr__(name, primal)
                return primal
            elif name == 'dual':
                return self._internal_var.getAttr('RC')
        else:
            super(Variable, self).__getattr__(name)
    
# class Constraint(interface.Constraint):
#     pass

# class Objective(interface.Objective):
#     pass

class Model(interface.Model):

    """docstring for Model"""
    _type2vtype = {'continuous': gurobipy.GRB.CONTINUOUS, 'integer': gurobipy.GRB.INTEGER, 'BINARY': gurobipy.GRB.BINARY}
    _vtype2type = dict([(val, key)for key, val in _type2vtype.iteritems()])
    _gurobi_status_to_status = {
        gurobipy.GRB.LOADED: interface.LOADED,
        gurobipy.GRB.OPTIMAL: interface.OPTIMAL,
        gurobipy.GRB.INFEASIBLE: interface.INFEASIBLE,
        gurobipy.GRB.INF_OR_UNBD: interface.INFEASIBLE_OR_UNBOUNDED,
        gurobipy.GRB.UNBOUNDED: interface.UNBOUNDED,
        gurobipy.GRB.CUTOFF: interface.CUTOFF,
        gurobipy.GRB.ITERATION_LIMIT: interface.ITERATION_LIMIT,
        gurobipy.GRB.NODE_LIMIT: interface.NODE_LIMIT,
        gurobipy.GRB.TIME_LIMIT: interface.TIME_LIMIT,
        gurobipy.GRB.SOLUTION_LIMIT: interface.SOLUTION_LIMIT,
        gurobipy.GRB.INTERRUPTED: interface.INTERRUPTED,
        gurobipy.GRB.NUMERIC: interface.NUMERIC,
        gurobipy.GRB.SUBOPTIMAL: interface.SUBOPTIMAL,
        gurobipy.GRB.INPROGRESS: interface.INPROGRESS
    }

    def __init__(self, problem=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        if problem is None:
            self.problem = gurobipy.Model()
        elif isinstance(problem, gurobipy.Model):
            self.problem = problem
            for gurobi_var in self.problem.getVars():
                var = Variable(
                            gurobi_var.getAttr("VarName"),
                            lb=gurobi_var.getAttr("lb"),
                            ub=gurobi_var.getAttr("ub"),
                            problem=self,
                            type=self._vtype2type[gurobi_var.getAttr("vType")]
                        )
               
                var._internal_var = gurobi_var
                super(Model, self)._add_variable(var)
            for constr in self.problem.getConstrs():
                super(Model, self)._add_constraint(
                        Constraint(
                            sympy.Rel(
                                sympy.var('x'),
                                constr.getAttr("RHS"),
                                self._gurobi_sense_to_sympy(constr.getAttr("Sense"))
                                ),
                            name=constr.getAttr("ConstrName"),
                            problem=self
                            )
                    )
        else:
            raise Exception("Provided problem is not a valid Gurobi model.")

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        self.problem.update()
        self.problem.write(tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def optimize(self):
        self.problem.optimize()
        self.status = self._gurobi_status_to_status[self.problem.getAttr("Status")]


    def _add_variable(self, variable):
        gurobi_var = self.problem.addVar(
            name=variable.name,
            lb=variable.lb,
            ub=variable.ub,
            vtype=self._type2vtype[variable.type]
        )
        variable._internal_var = gurobi_var
        super(Model, self)._add_variable(variable)

    def _remove_variable(self, variable):
        super(Model, self)._remove_variable(variable)
        if isinstance(variable, Variable):
            self.problem.remove(variable._internal_var)
        elif isinstance(variable, str):
            self.problem.remove(self.problem.getVarByName(variable))

    def _add_constraint(self, constraint):

        if constraint.expression.is_Add:
            lhs = gurobipy.LinExpr()
            lhs_terms = list()
            for arg in constraint.expression.args:
                if arg.is_Mul and len(arg.args) <= 2 and arg.args[0].is_Number:
                    coeff = float(arg.args[0])
                    if arg.args[1].is_Symbol:
                        var = arg.args[1]
                        if var.name not in self.variables:
                            self._add_variable(var)
                            gurobi_var = self.problem.addVar(name=var.name, lb=var.lb, ub=var.ub, vtype=self._type2vtype[var.type])
                            var._internal_var = gurobi_var
                            self.problem.update()
                        else:
                            gurobi_var = self.variables[var.name]._internal_var
                        lhs_terms.append(coeff * gurobi_var)
            lhs = gurobipy.quicksum(lhs_terms)
        else:
            raise Exception(' '.join(str(lhs), 'is not a sum of either linear or quadratic terms'))
            log.exception()

        if constraint.lb is constraint.ub:
            sense = '='
        elif constraint.lb and constraint.ub is None:
            sense = '>'
        elif constraint.lb is None and constraint.ub:
            sense = '<'
        elif constraint.lb is not None and constraint.ub is not None:
            sense = '='
            aux_var = self.problem.addVar(name=constraint.name+'_aux', lb=0, ub=constraint.ub - constraint.lb)
            self.problem.update()
            lhs -= aux_var
        if isinstance(lhs, gurobipy.LinExpr):
            self.problem.addConstr(lhs, sense, constraint.lb, name=constraint.name)
        elif isinstance(lhs, gurobipy.QuadExpr):
            self.problem.addQConstr(lhs, sense, rhs)
        super(Model, self)._add_constraint(constraint)

    def _sympy_relational_to_gurobi_constraint(self, constraint):
        if constraint.is_Relational:
            pass

    def _gurobi_sense_to_sympy(self, sense, translation={'=': '==', '<': '<', '>': '>'}):
        try:
            return translation[sense]
        except KeyError, e:
            print ' '.join('Sense', sense, 'is not a proper relational operator, e.g. >, <, == etc.')
            print e

if __name__ == '__main__':
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    z = Variable('z', lb=-100, ub=99.)
    constr = Constraint(0.3*x + 0.4*y + 66.*z, lb=-100, ub=0., name='test')

    from gurobipy import read
    solver = Model(problem=read("tests/data/model.lp"))
    
    solver.add(z)
    solver.add(constr)
    print solver
    print solver.optimize()
    print solver.status