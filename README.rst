|PyPI| |License| |Travis| |Coverage Status| |Code Climate|
|Documentation Status| |DOI|

optlang
=======

Vision
~~~~~~

**optlang** is a Python package for solving mathematical optimization
problems, i.e. maximizing or minimizing an objective function over a set
of variables subject to a number of constraints. Optlang provides a
common interface to a series of optimization tools, so different solver
backends can be changed in a transparent way.

Optlang takes advantage of the symbolic math library
`sympy <http://sympy.org/en/index.html>`__ to allow objective functions
and constraints to be easily formulated from symbolic expressions of
variables (see examples).

Installation
~~~~~~~~~~~~

Install using pip

::

    pip install optlang

Local installations like

::

    python setup.py install
     

might fail installing the dependencies (unresolved issue with
``easy_install``). Running

::

    pip install -r requirements.txt

beforehand should fix this issue.

Documentation
~~~~~~~~~~~~~

The documentation for **optlang** is provided at
`readthedocs.org <http://optlang.readthedocs.org/en/latest/>`__.

Dependencies
~~~~~~~~~~~~

-  `sympy >= 0.7.5 <http://sympy.org/en/index.html>`__
-  `six >= 1.9.0 <https://pypi.python.org/pypi/six>`__

And at least one of the following

-  `swiglpk >= 0.1.0 <https://pypi.python.org/pypi/swiglpk>`__
-  `cplex <https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`__
-  `gurobipy <http://www.gurobi.com>`__
-  `scipy <http://www.scipy.org>`__

Example
~~~~~~~

Formulating and solving the problem is straightforward (example taken
from `GLPK documentation <http://www.gnu.org/software/glpk>`__):

::

    from optlang.glpk_interface import Model, Variable, Constraint, Objective

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

The example will produce the following output:

::

    status: optimal
    objective value: 733.333333333
    x2 = 66.6666666667
    x3 = 0.0
    x1 = 33.3333333333

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

Requirements
~~~~~~~~~~~~

-  Models should always be serializable to common problem formulation
   languages
   (`CPLEX <http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`__,
   `GAMS <http://www.gams.com/>`__, etc.)
-  Models should be picklable
-  Common solver configuration interface (presolver, MILP gap, etc.)

.. |PyPI| image:: https://img.shields.io/pypi/v/optlang.svg?maxAge=2592000
   :target: https://pypi.python.org/pypi/optlang
.. |License| image:: http://img.shields.io/badge/license-APACHE2-blue.svg
   :target: http://img.shields.io/badge/license-APACHE2-blue.svg
.. |Travis| image:: https://img.shields.io/travis/biosustain/optlang/master.svg
   :target: https://travis-ci.org/biosustain/optlang
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/biosustain/optlang/master.svg
   :target: https://codecov.io/gh/biosustain/optlang/branch/master
.. |Code Climate| image:: https://codeclimate.com/github/biosustain/optlang/badges/gpa.svg
   :target: https://codeclimate.com/github/biosustain/optlang
.. |Documentation Status| image:: https://readthedocs.org/projects/optlang/badge/?version=latest
   :target: https://readthedocs.org/projects/optlang/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/5031/biosustain/optlang.svg
   :target: https://zenodo.org/badge/latestdoi/5031/biosustain/optlang
