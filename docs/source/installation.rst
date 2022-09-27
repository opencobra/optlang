Installation
============

Install optlang using pip::

  pip install optlang


Solvers
----------
To solve optimization problems, at least one supported solver must be installed.
Installing optlang using :code:`pip` will also automatically install GLPK. To use other solvers (e.g. commercial solvers) it is necessary
to install them manually. Optlang interfaces with all solvers through importable python modules. If the python module corresponding
to the solver can be imported without errors the solver interface should be available as an optlang submodule (e.g.
:code:`optlang.glpk_interface`).

The required python modules for the currently supported solvers are:

- GLPK: :code:`swiglpk` (automatically installed by :code:`pip install optlang`)

  - GLPK is an open source Linear Programming library. Swiglpk can be installed from binary wheels or from source. Installing from source requires swig and GLPK.

- Cplex: :code:`cplex`

  - Cplex is a very efficient commercial linear and quadratic mixed-integer solver from IBM. Academic licenses are available for students and researchers.

- Gurobi: :code:`gurobipy`

  - Gurobi is a very efficient commercial linear and quadratic mixed-integer solver. Academic licenses are available for students and researchers.

- SciPy: :code:`scipy.optimize.linprog`

  - The SciPy linprog function is a very basic implementation of the simplex algorithm for solving linear optimization problems. Linprog is included in all recent versions of SciPy.

- OSQP: :code:`osqp | cuosqp`

  - OSQP is an efficient open source solver for Quadratic Programs. It is self-contained and can be installed via pip. Alternatively, cuOSQP provides an experimental cuda-enabled implementation.

- Cbc: :code:`mip`

  - Cbc (Coin-or branch and cut) is an open-source mixed integer linear programming solver written in C++. It can be installed via mip (python3 only).


After importing optlang you can check :code:`optlang.available_solvers` to verify that a solver is recognized.


Issues
------

Local installations like

::

    python setup.py install


might fail installing the dependencies (unresolved issue with
``easy_install``). Running

::

    pip install -r requirements.txt

beforehand should fix this issue.
