=======

Next Release
------------

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
