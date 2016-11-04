Model
=====

The model object represents an optimization problem and contains the variables, constraints an objective that make up the problem. Variables and constraints can be added and removed using the :code:`.add` and :code:`.remove` methods, while the objective can be changed by setting the objective attribute, e.g. :code:`model.objective = Objective(expr, direction="max")`.

Once the problem has been formulated the optimization can be performed by calling the :code:`.optimize` method. This will return the status of the optimization, most commonly 'optimal', 'infeasible' or 'unbounded'.

.. autoclass:: optlang.interface.Model
  :members:
  :undoc-members:
  :show-inheritance:
