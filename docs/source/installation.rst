Installation
============

Install optlang using pip::

  pip install optlang

Or download the source distribution and run::

  python setup.py install

You can run optlang's test suite like this (you need to install nose first though)::

  python setup.py test
  
  
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
