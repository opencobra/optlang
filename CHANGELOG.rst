=======

Next Release
------------

1.6.0
-----
* fixes problem scaling for GLPK
* fixes scaling output in GLPK that could not be turned off
* Major performance improvements in the Gurobi interface when getting primals,
  shadow prices and reduced costs
* Now only supports gurobipy>=9.5
* Tests are now run with the Gurobi and CPLEX community editions
* Drop support for Python 2 and add support for Python 3.10

1.5.2
-----
* Gurobi can now serialize its configuration correctly. This also fixes pickling of Gurobi models.
* Fix the shim for integrality tolerance in OSQP which makes it easier to clone OSQP Models to other solvers.
* Fix an issue where one could not rename variables in Gurobi version 9.

1.5.1
-----
* GLPK now respects `Configuration.tolerances.integrality` again

1.5.0
-----
* removed support for Python 3.5
* added support for Python 3.9
* enabled code coverage in tox
* support symengine 0.7.0
* add [OSQP](https://github.com/oxfordcontrol/osqp) as additional solver
* add [cbc](https://github.com/coin-or/python-mip) as additional solver

1.4.7
-----
* fix: except AttributeError when setting tolerance on cloned Configuration
* enable cloning models between solver interfaces that do not support the same set of tolerance parameters
