---
title: 'Optlang: An algebraic modeling language for mathematical optimization'
tags:
 - Mathematical optimization
 - Linear programming
 - Object-oriented API
authors:
 - name: Kristian Jensen
   orcid: 0000-0002-2796-805X
   affiliation: 1
 - name: Joao G.R. Cardoso
   orcid: 0000-0001-8173-2673
   affiliation: 1
 - name: Nikolaus Sonnenschein
   orcid: 0000-0002-7581-4936
   affiliation: 1
affiliations:
 - name: The Novo Nordisk Foundation Center for Biosustainability, Technical University of Denmark
   index: 1
date: 6 December 2016
bibliography: paper.bib
---

# Summary

Optlang is a Python package implementing a modeling language for solving mathematical optimization problems, i.e., 
maximizing or minimizing an objective function over a set of variables subject to a number of constraints. It provides
a common native Python interface to a series of optimization tools, so different solver backends can be used and
changed in a transparent way.

Optlang's object-oriented API takes advantage of the symbolic math library SymPy [@Sympy] to allow objective functions
and constraints to be easily formulated algebraically from symbolic expressions of variables. Optlang targets
scientists who can thus focus on formulating optimization problems based on mathematical equations derived from domain
knowledge.

Solver interfaces can be added by subclassing the four main classes of the optlang API (Variable, Constraint, Objective,
and Model) and implementing the relevant API functions.


# References
