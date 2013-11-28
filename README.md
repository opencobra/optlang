optlang
=======

### Vision
__optlang__ provides a common interface to a series of optimization solvers (linear & non-linear) and relies on __sympy__ for problem formulation. Adding new solvers is easy: just subclass the high-level interface and implement solver specific routines.

### Version 0.1 roadmap

* Use _logging_ to provided access to solver log output
* Interfaces for gurobi and glpk

### Future outlook

* GAMS output
* Interface to NEOS


### Requirements
* An exception should be raised if a lower bound is set that is larger than an upper bound