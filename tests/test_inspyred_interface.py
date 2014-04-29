import unittest

import inspyred
from inspyred import benchmarks

from optlang.inspyred_interface import Model, Objective, Variable

# import logging
# logger = logging.getLogger('inspyred.ec')
# logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler('inspyred.log', mode='w')
# file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

def make_individual(evaluator):
    def _(candidate):
        return evaluator([candidate.values()], {})[0]

    return _


def my_observer(population, num_generations, num_evaluations, args):
    best = max(population)
    print('{0:6} -- {1} : {2}'.format(num_generations,
                                      best.fitness,
                                      str(best.candidate)))


class RosenbrockTestCase(unittest.TestCase):
    def setUp(self):
        benchmark = benchmarks.Rosenbrock()
        self.model = Model(algorithm='PSO')
        self.model.configuration.terminator = inspyred.ec.terminators.generation_termination
        x = Variable('x', lb=0, ub=2)
        y = Variable('y', lb=0, ub=2)

        rosenbrock_obj = Objective((1 - x) ** 2 + 100 * (y - y ** 2) ** 2, name="Rosenbrock function", direction='min')
        self.model.objective = rosenbrock_obj

    def test_evolutionary_strategy(self):
        self.model.configuration.max_generations = 100
        self.model.configuration.terminator = inspyred.ec.terminators.generation_termination
        self.model.configuration.observer = my_observer
        self.model.configuration.selector = inspyred.ec.selectors.tournament_selection
        final_pop = self.model.optimize()

        best = max(final_pop)
        self.assertAlmostEqual(best.fitness, 0)

    def test_pso(self):
        self.model.algorithm = 'PSO'
        self.model.configuration.max_generations = 100
        final_pop = self.model.optimize()

        best = max(final_pop)
        self.assertAlmostEqual(best.fitness, 0)



