---
title: 'Optlang: A Python interface to common mathematical optimization solvers'
tags:
 - Mathematical optimization
 - Linear programming
 - Object-oriented API
authors:
 -

affiliations:
 - name: Center for Biosustainability, Technical University of Denmark
   index: 1
date: 8 October 2016
bibliography: paper.bib
---

# Summary

Optlang is a Python package for solving mathematical optimization problems, i.e. maximizing or minimizing an
objective function over a set of variables subject to a number of constraints. It provides a common interface
to a series of optimization tools, so different solver backends can be changed in a transparent way.

Optlang takes advantage of the symbolic math library SymPy [@Sympy] to allow objective functions and constraints
to be easily formulated algebraically from symbolic expressions of variables. With Optlang the user can thus
focus on the science of formulating a problem without worrying about how to solve it.

Solver interfaces can be added by subclassing the 4 main classes of the Optlang API (Variable, Constraint, Objective
and Model) and implementing the relevant API functions.


# References