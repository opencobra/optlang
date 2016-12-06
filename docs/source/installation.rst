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
the solver interface should be available as an optlang submodule (e.g. :code:`optlang.glpk_interface`).

The required python modules for the currently supported solvers are:

- GLPK: :code:`swiglpk`

  - GLPK is an open source Linear Programming library. Swiglpk can be installed from binary wheels or from source. Installing from source requires swig and GLPK

- Cplex: :code:`cplex`

  - Cplex is a very efficient commercial linear and quadratic mixed-integer solver from IBM. Academic licenses are available for students and researchers.

- Gurobi: :code:`gurobipy`

  - Gurobi is a very efficient commercial linear and quadratic mixed-integer solver. Academic licenses are available for students and researchers.

- SciPy: :code:`scipy.optimize.linprog`

  - The SciPy linprog function is a very basic implementation of the simplex algorithm for solving linear optimization problems. Linprog is included in all recent versions of SciPy.

After importing optlang you can check :code:`optlang.available_solvers` to verify that a solver is recognized.
