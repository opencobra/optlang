---
title: 'Optlang: A Python interface to common mathematical optimization solvers'
tags:
 - Linear programming
 - Object-oriented API
 -
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

Optlang takes advantage of the symbolic math library sympy to allow objective functions and constraints
to be easily formulated from symbolic expressions of variables. With Optlang the user can focus on the science of
formulating a problem without worrying about how to solve it.