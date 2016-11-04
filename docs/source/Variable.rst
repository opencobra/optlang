Variable
========

Variable objects are used to represents each variable of the optimization problem. When the optimization is performed, the combination of variable values that optimizes the objective function, while not violating any constraints will be identified.
The type of a variable ('continuous', 'integer' or 'binary') can be set using the :code:`type` keyword of the constructor or it can be changed after initialization by :code:`var.type = 'binary'`.

The variable class subclasses the :code:`sympy.Symbol` class, which means that symbolic expressions of variables can be constructed by using regular python syntax, e.g. :code:`my_expression = 2 * var1 + 3 * var2 ** 2`. Expressions like this are used when constructing Constraint and Objective objects.

Once a problem has been optimized, the primal and dual values of a variable can be accessed from the :code:`primal` and :code:`dual` attributes, respectively.

.. autoclass:: optlang.interface.Variable
  :members:
  :undoc-members:
  :show-inheritance:
