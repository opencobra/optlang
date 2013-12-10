optlang development
===================

Contributions to optlang are very welcome. Fork optlang at `github
<http://github.com/phantomas1234/optlang>`_, implement your feature and send us
a pull request.

Add solver interface
--------------------

Put your new interface for solver *XYZ* into a file name xyz_interface.py and
put it into the optlang package folder.

Subclass :class:`interface.Model` like::
    
    class Model(interface.Model):
        def __init__(self, problem=None, *args, **kwargs):
            super(Model, self).__init__(*args, **kwargs)

:func:`time.time`
and then override the methods defined in :class:`interface.Model` to . For
example, override :func:`~interface.Model._add_constraint` to trigger solver XYZ's constraint
adding::

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

The existing solver interfaces can also be used as a reference.
