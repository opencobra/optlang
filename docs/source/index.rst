.. optlang documentation master file, created by
   sphinx-quickstart on Thu Nov 28 09:54:22 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

optlang
*******

|Build Status| |Coverage Status|

This package provides a generic interface to a series of optimization tools.
Currently supported solvers are:

* `GLPK <http://www.gnu.org/software/glpk/>`_ (via `Python-GLPK <http://www.dcc.fc.up.pt/~jpp/code/python-glpk/>`_)

Support for `CPLEX <http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`_, `GUROBI <http://www.gurobi.com/>`_, `MOSEK <http://www.mosek.com/>`_, and heuristic optiomization frameworks like `inspyred <https://pypi.python.org/pypi/inspyred>`_ are planned for future releases. **optlang** makes extensive use of the excellent symbolics library `SymPy <http://sympy.org>`_ to simplify problem formulation.


Installation
============

Install optlang using pip::

  pip optlang

Or, or download the source distribution and run::

  python setup.py install

You can run optlang's test suite like this::
  
  python setup.py test

Quick start
===========

Consider the following linear programming optimization problem (example taken from `GLPK documentation <http://www.gnu.org/software/glpk>`):

.. math::
    max\ 10 x_1 + 6 x_2 + 4 x_3

    subject\ to

    x_1 + x_2 + x_3 <= 100

    10 x_1 + 4 x_2 + 5 x_3 <= 600

    2 x_1 + 2 x_2 + 6 x_3 <= 300

    x_1 >= 0, x_2 >= 0, x_3 >= 0

Forumulating and solving the problem is straighforward::

  from optlang import Model, Variable, Constraint, Objective
  x1 = Variable('x1', lb=0)
  x2 = Variable('x2', lb=0)
  x3 = Variable('x3', lb=0)
  c1 = Constraint(x1 + x2 + x3, ub=100)
  c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)
  c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)
  obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
  model = Model(name='Simple model')
  model.objective = obj
  model.add([c1, c2, c3])
  status = model.optimize()
  print "status:", model.status
  print "objective value:", model.objective.value
  for var_name, var in model.variables.iteritems():
        print var_name, "=", var.primal

You should see the following output::

  status: optimal
  objective value: 733.333333333
  x2 = 66.6666666667
  x3 = 0.0
  x1 = 33.3333333333

Table of content
================

.. toctree::

   problem_formulation
   API
   .. optimization_and_solution_retrieval
   .. solver_parameters
   .. logging
   .. developers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Build Status| image:: https://travis-ci.org/biosustain/optlang.png?branch=master
   :target: https://travis-ci.org/biosustain/optlang
.. |Coverage Status| image:: https://coveralls.io/repos/biosustain/optlang/badge.png?branch=master
   :target: https://coveralls.io/r/biosustain/optlang?branch=master
.. |Coverage Status| image:: https://coveralls.io/repos/biosustain/optlang/badge.png?branch=master
   :target: https://coveralls.io/r/biosustain/optlang?branch=master

