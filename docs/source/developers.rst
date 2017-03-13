Contribute
==========

Contributions to optlang are very welcome. Fork optlang at `github
<http://github.com/biosustain/optlang>`_, implement your feature and send us
a pull request. Also, please use the GitHub `issue tracker <https://github.com/biosustain/optlang/issues>`_
to let us know about bugs or feature requests, or if you have problems or questions regarding optlang.

Add solver interface
--------------------

Put your interface for new solver *XYZ* into a python module with the
name xyz_interface.py. Please use the existing solver interfaces as a reference
for how to wrap a solver. For example, start by subclassing :class:`interface.Model`.

.. code-block:: python

    class Model(interface.Model):
        def __init__(self, problem=None, *args, **kwargs):
            super(Model, self).__init__(*args, **kwargs)

Then you can override the abstract methods defined in :class:`interface.Model`. For
example, override :func:`~interface.Model._add_constraints`.

.. code-block:: python

  def _add_constraint(self, constraint):
      # Check that constraint is actually supported by XYZ
      if not constraint.is_Linear:
          raise ValueError("XYZ only supports linear constraints. %s is not linear." % constraint)
      # Add the constraint to the user level interface
      super(Model, self)._add_constraint(constraint)
      # Add variables that are not yet in the model ...
      for var in constraint.variables:
          if var.name not in self.variables:
              self._add_variable(var)
      # Link the model to the constraint
      constraint.problem = self
      # Add solver specific code ...
      xyz_add_rows(self.problem, 1)
      index = xyz_get_num_rows(self.problem)
      xyz_set_row_name(self.problem, index, constraint.name)
      ...
