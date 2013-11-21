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
from interface import Solver
from gurobipy import Model


class GurobiSolver(Solver):

    """docstring for Solver"""

    def __init__(self, problem=None):
        if problem is None:
            self.problem = Model()
        elif isinstance(problem, Model):
            self.problem = problem
        else:
            raise Exception("Provided problem is not a valid Gurobi model.")

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        self.problem.update()
        self.problem.write(tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def _add_variable(self, variable):
        self.problem.addVar(
            name=variable.id, lb=variable.lb, ub=variable.ub, obj=variable.obj)

    def _remove_variable(self, variable):
        self.problem.remove(self.problem.getVarByName(variable.id))


    # @property
    # def bounds(self):
    #     bounds_dict = dict()
    #     for var in self.problem.getVars():
    #         bounds_dict[var.getAttr("VarName")] = (var.getAttr("lb"), var.getAttr("ub"))
    #     return bounds_dict
    # @bounds.setter
    # def bounds(self, bounds_dict):
    #     for key, val in bounds_dict:
    #         var = self.problem.getVarByName(key)
    #         var.setAttr("lb", val[0])
    #         var.setAttr("ub", val[1])


    # @property
    # def objective(self):
    #     return self.problem.getAttr()
    # @objective.setter
    # def objective(self, value):
    #     self.problem. = value
if __name__ == '__main__':
    from gurobipy import read
    solver = GurobiSolver(problem=read("model.lp"))
    print solver
