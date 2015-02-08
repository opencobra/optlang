# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Interface to the inspyred heuristic optimization framework

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""

import random
import logging

import types


log = logging.getLogger(__name__)
import sympy
import inspyred
import interface


class Variable(interface.Variable):
    def __init__(self, *args, **kwargs):
        super(Variable, self).__init__(*args, **kwargs)


class Objective(interface.Objective):
    """docstring for Objective"""

    def __init__(self, expression, *args, **kwargs):
        super(Objective, self).__init__(expression, *args, **kwargs)

    @property
    def value(self):
        return self._value

    def __str__(self):
        if isinstance(self.expression, sympy.Basic):
            return super(Objective, self).__str__()
        else:
            return self.expression.__str__()
            # return ' '.join((self.direction, str(self.expression)))

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = value


class VariableBounder(object):
    """This class defines a inspyred like Bounder.__init__.py

    TODO: Make this work also for integer and binary type variables?
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, candidate, args):
        variables = self.model.variables
        bounded_candidate = list()
        for c, variable in zip(candidate, variables):
            if variable.type == 'continuous':
                bounded_candidate.append(max(min(c, variable.ub), variable.lb))
            elif variable.type == 'integer':
                bounded_candidate.append(min(range(variable.lb, variable.ub + 1), key=lambda x: abs(x - c)))
            elif variable.type == 'binary':
                # print min([0, 1], key=lambda x: abs(x-c))
                bounded_candidate.append(min([0, 1], key=lambda x: abs(x - c)))
        return bounded_candidate


class Configuration(interface.EvolutionaryOptimizationConfiguration):
    """docstring for Configuration"""

    class SubConfiguration(object):
        pass

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self._algorithm = inspyred.ec.GA
        self._algorithm.terminator = [inspyred.ec.terminators.time_termination,
                                      inspyred.ec.terminators.generation_termination,
                                      inspyred.ec.terminators.evaluation_termination,
                                      inspyred.ec.terminators.diversity_termination,
                                      inspyred.ec.terminators.average_fitness_termination]
        self.pop_size = 100
        self.seeds = []
        self.max_generations = 1
        self.max_evaluations = None
        self.max_time = None

        self.selector_config = self.SubConfiguration()
        self.selector_config.num_selected = None
        self.selector_config.tournament_size = 2
        self.selector_config.num_elites = 0

        self.variator_config = self.SubConfiguration()
        self.variator_config.mutation_rate = .1
        self.variator_config.crossover_rate = 1.
        self.variator_config.num_crossover_points = 1

        self.topology_config = self.SubConfiguration()
        self.topology_config.neighborhood_size = 5

        self.swarm_config = self.SubConfiguration()
        self.swarm_config.inertia = 0.5
        self.swarm_config.cognitive_rate = 2.1
        self.swarm_config.social_rate = 2.1

    @property
    def selector(self):
        return self._algorithm.selector

    @selector.setter
    def selector(self, value):
        self.algorithm.selector = value

    @property
    def variator(self):
        return self._algorithm.variator

    @variator.setter
    def variator(self, value):
        self._algorithm.variator = value

    @property
    def replacer(self):
        return self._algorithm.replacer

    @replacer.setter
    def replacer(self, value):
        self._algorithm.replacer = value

    @property
    def migrator(self):
        return self._algorithm.migrator

    @migrator.setter
    def migrator(self, value):
        self._algorithm.migrator = value

    @property
    def archiver(self):
        return self._algorithm.archiver

    @archiver.setter
    def archiver(self, value):
        self._algorithm.archiver = value

    @property
    def observer(self):
        return self._algorithm.observer

    @observer.setter
    def observer(self, value):
        self._algorithm.observer = value

    @property
    def terminator(self):
        return self._algorithm.terminator

    @terminator.setter
    def terminator(self, value):
        self._algorithm.terminator = value

    @property
    def topology(self):
        return self._algorithm.topology

    @topology.setter
    def topology(self, value):
        if value == 'Ring':
            self._algorithm.topology = inspyred.swarm.topologies.ring_topology
        elif value == 'Star':
            self._algorithm.topology = inspyred.swarm.topologies.star_topology
        elif isinstance(value, types.FunctionType):
            self._algorithm.topology = value
        else:
            raise ValueError("%s is not a supported topology. Try 'Star' or 'Ring' instead.")

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        init = False
        try:
            previous_selector = self._algorithm.selector
            previous_variator = self._algorithm.variator
            previous_replacer = self._algorithm.replacer
            previous_migrator = self._algorithm.migrator
            previous_archiver = self._algorithm.archiver
            previous_observer = self._algorithm.observer
            previous_terminator = self._algorithm.terminator
        except AttributeError:
            init = True

        if value == "EvolutionaryComputation":
            self._algorithm = inspyred.ec.EvolutionaryComputation
        elif value == "GeneticAlgorithm" or value == "GA":
            self._algorithm = inspyred.ec.GA(random)
        elif value == "ParticleSwarmOptimization" or value == "PSO":
            self._algorithm = inspyred.swarm.PSO(random)
        elif value == "AntColonySystem" or value == "ACS":
            self._algorithm = inspyred.swarm.ACS(random)
        elif value == "EvolutionaryStrategy" or value == "ES":
            self._algorithm = inspyred.ec.ES(random)
        elif value == "DifferentialEvolutionaryAlgorithm" or value == "DEA":
            self._algorithm = inspyred.ec.DEA(random)
        elif value == "SimulatedAnnealing" or value == "SA":
            self._algorithm = inspyred.ec.SA(random)
        elif value == "NSGA2":
            self._algorithm = inspyred.emo.NSGA2(random)
        elif value == "PAES":
            self._algorithm = inspyred.emo.PAES(random)
        elif value == "Pareto":
            self._algorithm = inspyred.emo.Pareto(random)
        else:
            raise ValueError(
                "%s is not a supported. Try one of the following instead: 'GeneticAlgorithm', 'ParticleSwarmOptimization', 'EvolutionaryStrategy'. TODO: be more specific here")
        # self._algorithm.terminator = self._default_terminator
        if init is False:
            self._algorithm.selector = previous_selector
            self._algorithm.variator = previous_variator
            self._algorithm.replacer = previous_replacer
            previous_migrator = self._algorithm.migrator
            previous_archiver = self._algorithm.archiver
            previous_observer = self._algorithm.observer
            previous_terminator = self._algorithm.terminator
            # TODO: setting a new algorithm should recycle old variators, selectors etc.

    def _evolve_kwargs(self):
        """Filter None keyword arguments. Intended to be passed on to algorithm.evolve(...)"""
        valid_evolve_kwargs = (
            'max_generations', 'max_evaluations', 'pop_size', 'neighborhood_size', 'tournament_size', 'mutation_rate')
        filtered_evolve_kwargs = dict()
        for key in valid_evolve_kwargs:
            attr_value = getattr(self, key)
            if attr_value is not None:
                filtered_evolve_kwargs[key] = attr_value
        # return filtered_evolve_kwargs
        return {}


class Model(interface.Model):
    """Interface"""

    def __init__(self, algorithm=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.configuration = Configuration()
        if algorithm is None:
            # TODO: pick something smart?
            self.configuration.algorithm = "GA"
        else:
            self.configuration.algorithm = algorithm
        self._bounder = VariableBounder(self)
        self._generator = self._generator

    def _generator(self, random, args):
        individual = list()
        for variable in self.variables:
            if variable.type == 'continuous':
                individual.append(random.uniform(variable.lb, variable.ub))
            else:
                individual.append(random.choice(range(variable.lb, variable.ub + 1)))
        return individual

    def _evaluator(self, candidates, args):
        fitness = list()
        for candidate in candidates:
            substitution_dict = dict(zip(self.variables, candidate))
            if isinstance(self.objective.expression, sympy.Basic):
                fitness.append(self.objective.expression.subs(substitution_dict))
            else:
                fitness.append(self.objective.expression(substitution_dict))
        return fitness

    # @inspyred.ec.evaluators.evaluator
    # def _evaluate(self, candidate, args):
    #     substitution_dict = dict(zip(self.variables, candidate))
    #     try:
    #         fitness = self.objective.expression.subs(substitution_dict)
    #     except AttributeError:
    #         fitness = self.objective.expression(substitution_dict)
    #     return fitness

    def optimize(self, *args, **kwargs):
        # import pdb; pdb.set_trace();
        final_population = self.configuration.algorithm.evolve(
            generator=self._generator,
            evaluator=self._evaluator,
            bounder=self._bounder,
            pop_size=self.configuration.pop_size,
            maximize={'max': True, 'min': False}[self.objective.direction],
            max_generations=self.configuration.max_generations,
            max_evaluations=self.configuration.max_evaluations,
            neighborhood_size=self.configuration.topology_config.neighborhood_size,
            mutation_rate=self.configuration.variator_config.mutation_rate,
            tournament_size=self.configuration.selector_config.tournament_size
        )
        return final_population


if __name__ == '__main__':
    # from optlang.interface import Objective, Variable
    import numpy
    import inspyred

    x = Variable('x', lb=0, ub=2)
    y = Variable('y', lb=0, ub=2)
    rosenbrock_obj = Objective((1 - x) ** 2 + 100 * (y - x ** 2) ** 2, name="Rosenbrock function", direction='min')
    print("The rosenbrock function:", rosenbrock_obj)
    print("The global minimum at (x,y) = (1,1) is", rosenbrock_obj.expression.subs({x: 1, y: 1}))

    problem = Model(name='rosenbrock', algorithm='PSO')
    # problem = Model(name='rosenbrock')

    problem.objective = rosenbrock_obj

    def my_observer(population, num_generations, num_evaluations, args):
        best = max(population)
        print(('{0:6} -- {1} : {2}'.format(num_generations,
                                          best.fitness,
                                          str(best.candidate))))

    problem.configuration.max_generations = 100
    problem.configuration.terminator = inspyred.ec.terminators.generation_termination
    problem.configuration.observer = my_observer
    problem.configuration.selector = inspyred.ec.selectors.tournament_selection
    final_pop = problem.optimize()
    fitnesses = [individual.fitness for individual in final_pop]
    print(fitnesses)
    print("mean", numpy.mean(fitnesses))
    print("max", numpy.max(fitnesses))
    print("min", numpy.min(fitnesses))
    # print numpy.std(fitnesses)

