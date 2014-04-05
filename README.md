[![Build Status](https://travis-ci.org/biosustain/optlang.png?branch=master)](https://travis-ci.org/biosustain/optlang)
[![Coverage Status](https://coveralls.io/repos/biosustain/optlang/badge.png?branch=master)](https://coveralls.io/r/biosustain/optlang?branch=master)
[![PyPI version](https://badge.fury.io/py/optlang.svg)](http://badge.fury.io/py/optlang)

optlang
=======

### Vision
__optlang__ provides a common interface to a series of optimization solvers (linear & non-linear) and relies on [sympy](http://sympy.org/en/index.html) for problem formulation (constraints, objectives, variables, etc.). Adding new solvers is easy: just sub-class the high-level interface and implement the necessary solver specific routines.

### Documentation

The documentation for __optlang__ is provided at [readthedocs.org](http://optlang.readthedocs.org/en/latest/).

### Version 0.1 roadmap

- [ ] Interfaces for GLPK (reference implementation) and [CPLEX][cplex_url] (basic support)
- [ ] Use _logging_ to provide a common interface to solver log output
- [ ] Reach >90% test coverage (for GLPK interface)
- [ ] Documentation on [readthedocs.org](http://readthedocs.org)

### Future outlook

* [Gurobi][gurobi_url] interface (very efficient MILP solver)
* [Mosek][mosek_url] interface (provides academic licenses)
* [GAMS][gams_url] output (support non-linear problem formulation)
* [DEAP][deap_url] (support for heuristic optimization)
* Interface to [NEOS][neos_url] optimization server (for testing purposes and solver evaluation)
* Automatically handle fractional and absolute value problems when dealing with LP/MILP/QP solvers (like GLPK, [CPLEX][cplex_url] etc.)

### Requirements

* Models should always be serializable to common problem formulation languages ([CPLEX][cplex_url], [GAMS][gams_url], etc.)
* Models should be pickable
* Common solver configuration interface (presover, MILP gap, etc.)

### Notes

Supporting heuristic optimization too? Only objectives and variables would be needed and constraints would be superfluous. Objectives would probably have to support non-mathematical evaluation functions.

Objective and Constraint could probably inherit from a common base class.

use `pyreverse -my -o pdf optlang` to generate a UML diagram

Looks like it is [not ok](https://code.google.com/p/sympy/issues/detail?id=3680#c7) to change sympy Symbol's name attribute (doesn't raise an error though).
Checkout [http://www.cuter.rl.ac.uk/](http://www.cuter.rl.ac.uk/)
Should Variables be singletons?

[cplex_url]: http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/ "CPLEX"
[gurobi_url]: http://www.gurobi.com/  "GUROBI"
[mosek_url]: http://www.mosek.com/ "MOSEK"
[gams_url]: http://www.gams.com/ "GAMS"
[deap_url]: https://code.google.com/p/deap/ "DEAP"
[neos_url]: http://www.neos-server.org/neos/ "NEOS"

