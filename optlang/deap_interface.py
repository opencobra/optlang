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


"""Interface to the DEAP heuristic optimization framework 

http://nbviewer.ipython.org/github/DEAP/notebooks/blob/master/SIGEvolution.ipynb

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""

import copy
import random
import logging
log = logging.getLogger(__name__)
import tempfile
import sympy
import interface
import deap

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(model.reactions))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)





if __name__ == '__main__':
    # from optlang.interface import Objective, Variable
    import numpy

    x = Variable('x', lb=.8, ub=1.2)
    y = Variable('y', lb=.8, ub=1.2)
    rosenbrock = Objective((1 - x)**2 + 100 * (y - y**2)**2, name="Rosenbrock function", direction='min')
    print "The rosenbrock function:", rosenbrock
    print "The global minimum at (x,y) = (1,1) is", rosenbrock.expression.subs({x: 1, y: 1})

    rosenbrock_problem = Model(name='rosenbrock', algorithm='PSO')
    rosenbrock_problem.add([x, y])
    rosenbrock_problem.objective = rosenbrock
    final_pop = rosenbrock_problem.optimize()
    fitnesses = [individual.fitness for individual in final_pop]
    print fitnesses
    print "mean", numpy.mean(fitnesses)
    print "max", numpy.max(fitnesses)
    print "min", numpy.min(fitnesses)
    # print numpy.std(fitnesses)


