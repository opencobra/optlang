optlang
*******

Optlang is a Python package implementing a modeling language for solving mathematical optimization problems, i.e. maximizing or minimizing an objective function over a set of variables subject to a number of constraints.
Optlang provides a common interface to a series of optimization tools, so different solver backends can be changed in a transparent way.

In constrast to e.g. the commonly used General Algebraic Modeling System (GAMS), optlang has a simple and intuitive interface using native Python algebra syntax, and is free and open-source.

Optlang takes advantage of the symbolic math library `SymPy <http://sympy.org>`_ to allow objective functions and constraints to be easily formulated from symbolic expressions of variables (see examples).
Scientists can thus use optlang to formulate their optimization problems using mathematical expressions derived from domain knowledge.

Currently supported solvers are:

* `GLPK <http://www.gnu.org/software/glpk/>`_ (LP/MILP; via `swiglpk <https://github.com/biosustain/swiglpk>`_)
* `CPLEX <http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/>`_ (LP/MILP/QP)
* `Gurobi <http://www.gurobi.com/>`_ (LP/MILP/QP)
* `inspyred <https://pypi.python.org/pypi/inspyred>`_ (heuristic optimization; experimental)

Support for the following solvers is in the works:

* `GAMS <http://www.gurobi.com/>`_ (LP/MILP/QP/NLP; will include support for solving problems on `neos-server.org <https://neos-server.org/neos/>`_)
* `SOPLEX <http://soplex.zib.de>`_ (exact LP)
* `MOSEK <http://www.mosek.com/>`_, (LP/MILP/QP)



Quick start
===========

Consider the following linear programming optimization problem (example taken from `GLPK documentation <http://www.gnu.org/software/glpk>`_):

.. math::
    \begin{aligned}
        Max~ & ~ 10 x_1 + 6 x_2 + 4 x_3 \\
        s.t.~ & ~ x_1 + x_2 + x_3 <= 100 \\
        ~ & ~ 10 x_1 + 4 x_2 + 5 x_3 <= 600 \\
        ~ & ~ 2 x_1 + 2 x_2 + 6 x_3 <= 300 \\
        ~ & ~ x_1 \geq 0, x_2 \geq 0, x_3 \geq 0
    \end{aligned}




Formulating and solving the problem is straightforward

.. code-block:: python

   from optlang import Model, Variable, Constraint, Objective
   
   # All the (symbolic) variables are declared, with a name and optionally a lower and/or upper bound.
   x1 = Variable('x1', lb=0)
   x2 = Variable('x2', lb=0)
   x3 = Variable('x3', lb=0)
   
   # A constraint is constructed from an expression of variables and a lower and/or upper bound (lb and ub).
   c1 = Constraint(x1 + x2 + x3, ub=100)
   c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)
   c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)
   
   # An objective can be formulated
   obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
   
   # Variables, constraints and objective are combined in a Model object, which can subsequently be optimized.
   model = Model(name='Simple model')
   model.objective = obj
   model.add([c1, c2, c3])
   status = model.optimize()
   print("status:", model.status)
   print("objective value:", model.objective.value)
   print("----------")
   for var_name, var in model.variables.items():
       print(var_name, "=", var.primal)

You should see the following output::

  status: optimal
  objective value: 733.333333333
  ----------
  x2 = 66.6666666667
  x3 = 0.0
  x1 = 33.3333333333


Using a particular solver
----------
If you have more than one solver installed, it's also possible to specify which one to use, by importing directly from the
respective solver interface, e.g. :code:`from optlang.glpk_interface import Model, Variable, Constraint, Objective`

Quadratic programming
----------
A QP problem can be generated in the same way by creating an objective with a quadratic expression. In the above example
the objective could be :code:`obj = Objective(x1 ** 2 + x2 ** 2 - 10 * x1, direction="min")` to specify a quadratic minimization
problem.

Integer programming
----------
Integer (or mixed integer) problems can be specified by assigning the type of one or more variables to 'integer' or 'binary'.
If the solver supports integer problems it will automatically use the relevant optimization algorithm and return an integer solution.

Example
============

The GAMS example (http://www.gams.com/docs/example.htm) can be formulated and solved in optlang like this:

.. code-block:: python

    from optlang import Variable, Constraint, Objective, Model

    # Define problem parameters
    # Note this can be done using any of Python's data types. Here we have chosen dictionaries
    supply = {"Seattle": 350, "San_Diego": 600}
    demand = {"New_York": 325, "Chicago": 300, "Topeka": 275}

    distances = {  # Distances between locations in thousands of miles
        "Seattle": {"New_York": 2.5, "Chicago": 1.7, "Topeka": 1.8},
        "San_Diego": {"New_York": 2.5, "Chicago": 1.8, "Topeka": 1.4}
    }

    freight_cost = 9  # Cost per case per thousand miles

    # Define variables
    variables = {}
    for origin in supply:
        variables[origin] = {}
        for destination in demand:
            # Construct a variable with a name, bounds and type
            var = Variable(name="{}_to_{}".format(origin, destination), lb=0, type="integer")
            variables[origin][destination] = var

    # Define constraints
    constraints = []
    for origin in supply:
        const = Constraint(
            sum(variables[origin].values()),
            ub=supply[origin],
            name="{}_supply".format(origin)
        )
        constraints.append(const)
    for destination in demand:
        const = Constraint(
            sum(row[destination] for row in variables.values()),
            lb=demand[destination],
            name="{}_demand".format(destination)
        )
        constraints.append(const)

    # Define the objective
    obj = Objective(
        sum(freight_cost * distances[ori][dest] * variables[ori][dest] for ori in supply for dest in demand),
        direction="min"
    )
    # We can print the objective and constraints
    print(obj)
    print("")
    for const in constraints:
        print(const)

    print("")

    # Put everything together in a Model
    model = Model()
    model.add(constraints)  # Variables are added implicitly
    model.objective = obj

    # Optimize and print the solution
    status = model.optimize()
    print("Status:", status)
    print("Objective value:", model.objective.value)
    print("")
    for var in model.variables:
        print(var.name, ":", var.primal)
        
Outputting the following::

    Minimize
    16.2*San_Diego_to_Chicago + 22.5*San_Diego_to_New_York + 12.6*San_Diego_to_Topeka + 15.3*Seattle_to_Chicago + 22.5*Seattle_to_New_York + 16.2*Seattle_to_Topeka

    Seattle_supply: Seattle_to_Chicago + Seattle_to_New_York + Seattle_to_Topeka <= 350
    San_Diego_supply: San_Diego_to_Chicago + San_Diego_to_New_York + San_Diego_to_Topeka <= 600
    Chicago_demand: 300 <= San_Diego_to_Chicago + Seattle_to_Chicago
    Topeka_demand: 275 <= San_Diego_to_Topeka + Seattle_to_Topeka
    New_York_demand: 325 <= San_Diego_to_New_York + Seattle_to_New_York

    Status: optimal
    Objective value: 15367.5

    Seattle_to_New_York : 50
    Seattle_to_Chicago : 300
    Seattle_to_Topeka : 0
    San_Diego_to_Chicago : 0
    San_Diego_to_Topeka : 275
    San_Diego_to_New_York : 275
    
Here we forced all variables to have integer values. To allow non-integer values, leave out :code:`type="integer"` in the Variable constructor (defaults to :code:`'continuous'`).

Users's guide
=============

.. toctree::

   installation
   developers
   API
   .. problem_formulation
.. optimization_and_solution_retrieval
.. solver_parameters
.. logging

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
