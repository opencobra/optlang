optlang
=======

*Sympy based mathematical programming language*

|PyPI| |License| |Travis| |Appveyor| |Coverage Status| |Documentation Status| |DOI|

Optlang is a Python package for solving mathematical optimization
problems, i.e. maximizing or minimizing an objective function over a set
of variables subject to a number of constraints. Optlang provides a
common interface to a series of optimization tools, so different solver
backends can be changed in a transparent way.
Optlang's object-oriented API takes advantage of the symbolic math library
`sympy <http://sympy.org/en/index.html>`__ to allow objective functions
and constraints to be easily formulated from symbolic expressions of
variables (see examples).

Show us some love by staring this repo if you find optlang useful!

Also, please use the GitHub `issue tracker <https://github.com/biosustain/optlang/issues>`_
to let us know about bugs or feature requests, or if you have problems or questions regarding optlang.

Installation
~~~~~~~~~~~~

Install using pip

::

    pip install optlang

This will also install `swiglpk <https://github.com/biosustain/swiglpk>`_, an interface to the open source (mixed integer) LP solver `GLPK <https://www.gnu.org/software/glpk/>`_.
Quadratic programming (and MIQP) is supported through additional optional solvers (see below).

Dependencies
~~~~~~~~~~~~

The following dependencies are needed.

-  `sympy >= 1.0.0 <http://sympy.org/en/index.html>`__
-  `six >= 1.9.0 <https://pypi.python.org/pypi/six>`__
-  `swiglpk >= 1.3.0 <https://pypi.python.org/pypi/swiglpk>`__

The following are optional dependencies that allow other solvers to be used.

-  `cplex <https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`__ (LP, MILP, QP, MIQP)
-  `gurobipy <http://www.gurobi.com>`__ (LP, MILP (QP and MIQP support will be added in the future))
-  `scipy <http://www.scipy.org>`__ (LP)



Example
~~~~~~~

Formulating and solving the problem is straightforward (example taken
from `GLPK documentation <http://www.gnu.org/software/glpk>`__):

.. code-block:: python

    from __future__ import print_function
    from optlang import Model, Variable, Constraint, Objective

    # All the (symbolic) variables are declared, with a name and optionally a lower and/or upper bound.
    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)

    # A constraint is constructed from an expression of variables and a lower and/or upper bound (lb and ub).
    c1 = Constraint(x1 + x2 + x3, ub=100)
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)

    # An objective can be formulated
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')

    # Variables, constraints and objective are combined in a Model object, which can subsequently be optimized.
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])

    status = model.optimize()

    print("status:", model.status)
    print("objective value:", model.objective.value)
    print("----------")
    for var_name, var in model.variables.iteritems():
        print(var_name, "=", var.primal)

The example will produce the following output:

::

    status: optimal
    objective value: 733.333333333
    ----------
    x2 = 66.6666666667
    x3 = 0.0
    x1 = 33.3333333333

Using a particular solver
-------------------------
If you have more than one solver installed, it's also possible to specify which one to use, by importing directly from the
respective solver interface, e.g. :code:`from optlang.glpk_interface import Model, Variable, Constraint, Objective`

Documentation
~~~~~~~~~~~~~

Documentation for optlang is provided at
`readthedocs.org <http://optlang.readthedocs.org/en/latest/>`__.


Future outlook
~~~~~~~~~~~~~~

-  `Mosek <http://www.mosek.com/>`__ interface (provides academic
   licenses)
-  `GAMS <http://www.gams.com/>`__ output (support non-linear problem
   formulation)
-  `DEAP <https://code.google.com/p/deap/>`__ (support for heuristic
   optimization)
-  Interface to `NEOS <http://www.neos-server.org/neos/>`__ optimization
   server (for testing purposes and solver evaluation)
-  Automatically handle fractional and absolute value problems when
   dealing with LP/MILP/QP solvers (like GLPK,
   `CPLEX <http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`__
   etc.)

The optlang `trello board <https://trello.com/b/aiwfbVKO/optlang>`__
also provides a good overview of the project's roadmap.

.. |PyPI| image:: https://img.shields.io/pypi/v/optlang.svg?maxAge=2592000
   :target: https://pypi.python.org/pypi/optlang
.. |License| image:: http://img.shields.io/badge/license-APACHE2-blue.svg
   :target: http://img.shields.io/badge/license-APACHE2-blue.svg
.. |Travis| image:: https://img.shields.io/travis/biosustain/optlang/master.svg
   :target: https://travis-ci.org/biosustain/optlang
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/biosustain/optlang/master.svg
   :target: https://codecov.io/gh/biosustain/optlang/branch/master
.. |Documentation Status| image:: https://readthedocs.org/projects/optlang/badge/?version=latest
   :target: https://readthedocs.org/projects/optlang/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/5031/biosustain/optlang.svg
   :target: https://zenodo.org/badge/latestdoi/5031/biosustain/optlang
.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/443yp8hf25c6748h/branch/master?svg=true
   :target: https://ci.appveyor.com/project/phantomas1234/optlang/branch/master

