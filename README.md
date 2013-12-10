optlang
=======

### Vision
__optlang__ provides a common interface to a series of optimization solvers (linear & non-linear) and relies on __sympy__ for problem formulation. Adding new solvers is easy: just subclass the high-level interface and implement solver specific routines.

### Version 0.1 roadmap

* Use _logging_ to provided access to solver log output
* Interfaces for gurobi, mosek, cplex, and glpk

### Future outlook

* GAMS output
* Interface to NEOS
* Automatically handle fractional and absolute value problems when dealing with solvers like GLPK, CPLEX etc.

### Requirements
An exception should be raised if a lower bound is set that is larger than the currently set upper bound.

With statement with model creation e.g.

	with Model(param ...) as m:
		m.add(constr)
		m.optimize()

### Notes

Looks like is is [not ok](https://code.google.com/p/sympy/issues/detail?id=3680#c7) to change sympy Symbol's name attribute (doesn't raise an error though).
Checkout http://www.cuter.rl.ac.uk/
Should Variables be singletons?
	    
	    