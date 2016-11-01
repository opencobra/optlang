optlang
*******

Optlang is a Python package for solving mathematical optimization problems, i.e. maximizing or minimizing an objective function over a set of variables subject to a number of constraints.
Optlang provides a common interface to a series of optimization tools, so different solver backends can be changed in a transparent way.

Optlang takes advantage of the symbolic math library `SymPy <http://sympy.org>`_ to allow objective functions and constraints to be easily formulated from symbolic expressions of variables (see examples).

Currently supported solvers are:

* `GLPK <http://www.gnu.org/software/glpk/>`_ (LP/MILP; via `swiglpk <https://github.com/biosustain/swiglpk>`_)
* `CPLEX <http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`_ (LP/MILP/QP)
* `Gurobi <http://www.gurobi.com/>`_ (LP/MILP/QP)
* `inspyred <https://pypi.python.org/pypi/inspyred>`_ (heuristic optimization; experimental)

Support for the following solvers is in the works:

* `GAMS <http://www.gurobi.com/>`_ (LP/MILP/QP/NLP; will included support for solving problems on `neos-server.org <https://neos-server.org/neos/>`_)
* `SOPLEX <http://soplex.zib.de>`_ (exact LP)
* `MOSEK <http://www.mosek.com/>`_, (LP/MILP/QP)



Quick start
===========

Consider the following linear programming optimization problem (example taken from `GLPK documentation <http://www.gnu.org/software/glpk>`_):

.. math::
    \begin{aligned}
        Max~ & ~ 10 x_1 + 6 x_2 + 4 x_3 \\
        s.t.~ & ~ x_1 + x_2 + x_3 <= 100 \\
        ~ & ~ 10 x_1 + 4 x_2 + 5 x_3 <= 600 \\
        ~ & ~ 2 x_1 + 2 x_2 + 6 x_3 <= 300 \\
        ~ & ~ x_1 \geq 0, x_2 \geq 0, x_3 \geq 0
    \end{aligned}




Formulating and solving the problem is straightforward

.. code-block:: python

   from optlang import Model, Variable, Constraint, Objective
   
   # All the (symbolic) variables are declared, with a name and optionally a lower and/or upper bound.
   x1 = Variable('x1', lb=0)
   x2 = Variable('x2', lb=0)
   x3 = Variable('x3', lb=0)
   
   # A constraint is constructed from an expression of variables and a lower and/or upper bound (lb and ub).
   c1 = Constraint(x1 + x2 + x3, ub=100)
   c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)
   c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)
   
   # An objective 
   obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
   
   # Variables, constraints and objective are combined in a Model object, which can subsequently be optimized.
   model = Model(name='Simple model')
   model.objective = obj
   model.add([c1, c2, c3])
   status = model.optimize()
   print("status:", model.status)
   print("objective value:", model.objective.value)
   print("----------")
   for var_name, var in model.variables.items():
       print(var_name, "=", var.primal)

You should see the following output::

  status: optimal
  objective value: 733.333333333
  ----------
  x2 = 66.6666666667
  x3 = 0.0
  x1 = 33.3333333333

Users's guide
=============

.. toctree::

   installation
   developers
   API
   .. problem_formulation
.. optimization_and_solution_retrieval
.. solver_parameters
.. logging

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
