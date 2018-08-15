import numpy as np
from optlang import Model, Variable, Constraint, Objective

# All the (symbolic) variables are declared, with a name and optionally a lower
# and/or upper bound.
x = np.array([Variable('x{}'.format(i), lb=0) for i in range(1, 4)])

bounds = [100, 600, 300]

A = np.array([[1, 1, 1],
              [10, 4, 5],
              [2, 2, 6]])

w = np.array([10, 6, 4])

obj = Objective(w.dot(x), direction='max')

c = np.array([Constraint(row, ub=bound) for row, bound in zip(A.dot(x), bounds)])

model = Model(name='Numpy model')
model.objective = obj
model.add(c)

status = model.optimize()

print("status:", model.status)
print("objective value:", model.objective.value)
print("----------")
for var_name, var in model.variables.iteritems():
    print(var_name, "=", var.primal)
