# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

# from unittest import TestCase
# from optlang import interface, glpk_interface, gurobi_interface


# class TestSolver(TestCase):

#     def setUp(self):

#         self.solver = Solver()
#         x = Variable('x', lb=0, ub=10)
#         y = Variable('y', lb=0, ub=10)
#         constr = Constraint(x + y > 3, name="constr1")
#         obj = Objective(2*x + y)
#         self.solver.add(x)
#         self.solver.add(y)
#         self.solver.add(obj)

#     def test_1(self):
#         self.assertEqual(1+1, 2)

#     def test_solve(self):
#         sol = solver.optimize()
#         self.assertEqual(sol, 3.)