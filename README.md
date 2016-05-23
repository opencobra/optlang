[![PyPI](https://img.shields.io/pypi/v/optlang.svg?maxAge=2592000)](https://pypi.python.org/pypi/optlang)
[![License](http://img.shields.io/badge/license-APACHE2-blue.svg)](http://img.shields.io/badge/license-APACHE2-blue.svg)
[![Travis](https://img.shields.io/travis/biosustain/optlang/master.svg)](https://travis-ci.org/biosustain/optlang)
[![Coverage Status](https://img.shields.io/codecov/c/github/biosustain/optlang/master.svg)](https://codecov.io/gh/biosustain/optlang/branch/master)
[![Code Climate](https://codeclimate.com/github/biosustain/optlang/badges/gpa.svg)](https://codeclimate.com/github/biosustain/optlang)
[![Documentation Status](https://readthedocs.org/projects/optlang/badge/?version=latest)](https://readthedocs.org/projects/optlang/?badge=latest)



optlang
=======

### Vision
__optlang__ provides a common interface to a series of optimization solvers (linear & non-linear) and relies on [sympy](http://sympy.org/en/index.html) for problem formulation (constraints, objectives, variables, etc.). Adding new solvers is easy: just sub-class the high-level interface and implement the necessary solver specific routines.

### Installation

Install using pip
    
    pip install optlang
    
Local installations like
    
    python setup.py install
     
might fail installing the dependencies (unresolved issue with `easy_install`). Running
 
    pip install -r requirements.txt

beforehand should fix this issue.

### Documentation

The documentation for __optlang__ is provided at [readthedocs.org](http://optlang.readthedocs.org/en/latest/).

### Dependencies

* [sympy >= 0.7.5](http://sympy.org/en/index.html)
* [swiglpk >= 0.1.0](https://pypi.python.org/pypi/swiglpk)
* [glpk >= 4.45](https://www.gnu.org/software/glpk/)

### Example

Formulating and solving the problem is straightforward (example taken from [GLPK documentation](http://www.gnu.org/software/glpk)):

    from optlang import Model, Variable, Constraint, Objective
 
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
 
    status: optimal
    objective value: 733.333333333
    x2 = 66.6666666667
    x3 = 0.0
    x1 = 33.3333333333
    
### Future outlook

* [Gurobi][gurobi_url] interface (very efficient MILP solver)
* [CPLEX][cplex_url] interface (very efficient MILP solver)
* [Mosek][mosek_url] interface (provides academic licenses)
* [GAMS][gams_url] output (support non-linear problem formulation)
* [DEAP][deap_url] (support for heuristic optimization)
* Interface to [NEOS][neos_url] optimization server (for testing purposes and solver evaluation)
* Automatically handle fractional and absolute value problems when dealing with LP/MILP/QP solvers (like GLPK, [CPLEX][cplex_url] etc.)

The optlang [trello board](https://trello.com/b/aiwfbVKO/optlang) also provides a good overview of the project's roadmap.

### Requirements

* Models should always be serializable to common problem formulation languages ([CPLEX][cplex_url], [GAMS][gams_url], etc.)
* Models should be pickable
* Common solver configuration interface (presolver, MILP gap, etc.)

[cplex_url]: http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/ "CPLEX"
[inspyred_url]: https://pypi.python.org/pypi/inspyred
[gurobi_url]: http://www.gurobi.com/  "GUROBI"
[mosek_url]: http://www.mosek.com/ "MOSEK"
[gams_url]: http://www.gams.com/ "GAMS"
[deap_url]: https://code.google.com/p/deap/ "DEAP"
[neos_url]: http://www.neos-server.org/neos/ "NEOS"

