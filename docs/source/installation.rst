Installation
============

Install optlang using pip::

  pip install optlang

Or, or download the source distribution and run::

  python setup.py install

You can run optlang's test suite like this (you need to install nose first though)::

  python setup.py test
  
  
Solvers
----------
In addition to optlang itself, it is necessary to install at least one solver. Optlang interfaces with the solvers
through importable python modules. If the python module corresponding to the solver can be imported without errors
the solver should be available as an Optlang interface.

The required python modules for the currently supported solvers are:

- GLPK: :code:`swglpk`
- Cplex: :code:`cplex`
- Gurobi: :code:`gurobipy`
- Scipy: :code:`scipy.optimize.linprog`

You can call the function :code:`optlang.list_available_solvers` to verify that a solver is recognized by optlang.
