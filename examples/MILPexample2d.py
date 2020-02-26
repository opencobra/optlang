import matplotlib.pyplot as plt
import numpy as np
from optlang import Model, Variable, Constraint, Objective

from plot_opt_2d import region_plot, plot_LP

x = Variable('x',lb=0,type='integer')
y = Variable('y',lb=0,type='continuous')

xmin, xmax = -1,4
ymin, ymax = -1,4

constraints = [
    Constraint(4*x+4*y, lb=1, ub=7),
    Constraint(4*x-4*y, lb=-3, ub=3)
]
obj = Objective(x+y, direction='max')

model = Model(name='My model')
model.objective = obj
model.add(constraints)

status = model.optimize()

print("status:", model.status)
print("objective value:", model.objective.value)

for var_name, var in model.variables.iteritems():
    print(var_name, "=", var.primal)

plt.figure(figsize=(6,6))
plot_LP(model, (x,-1,4), (y,-1,4))
plt.show()
