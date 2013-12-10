.. optlang documentation master file, created by
   sphinx-quickstart on Thu Nov 28 09:54:22 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

optlang - Generic optimization solver interface
***********************************************

This package provides a generic interface to a series of optimization tools.
Currently supported solvers are:

* `GLPK <http://www.gnu.org/software/glpk/>`_ (via `Python-GLPK <http://www.dcc.fc.up.pt/~jpp/code/python-glpk/>`_)
* `GUROBI <http://www.gurobi.com>`_ (via its own python bindings)

Support for CPLEX and MOSEK are planned for future releases.

Design considerations
=====================

**optlang** makes use of the Python computer algebra system `SymPy <http://sympy.org>`_. 


Installation and quick start
============================

Install optlang using pip::

  pip optlang

Or, or download the source distribution and run::

  python setup.py install

You can run optlang's test suite like this::
  
  python setup.py test

Now you can solve optimization problems like::

  from optlang import Model, Variable, Constraint, Objective
  model = Model()
  x = Variable('x', lb=0, ub=10)
  y = Variable('y', lb=0, ub=10)
  constr = Constraint(x + y, lb=3, name="constr1")
  obj = Objective(2 * x + y)
  model.add(constr)
  model.add(obj)
  model.optimization()

You should see the following output::

  ....  


.. toctree::

   problem_formulation
   optimization_and_solution_retrieval
   solver_parameters
   logging
   developers
   API


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

