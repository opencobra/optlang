# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

from optlang import Model, Variable, Constraint

model = Model()
x = Variable('x', lb=0, ub=10)
y = Variable('y', lb=0, ub=10)
constr = Constraint(x + y, lb=3, name="constr1")
obj = Objective(2 * x + y)

model.add(constr)
model.add(obj)
model.optimization()
for var in model.variables:
	
