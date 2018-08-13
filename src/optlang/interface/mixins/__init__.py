# -*- coding: utf-8 -*-

# Copyright 2013-2017 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provide mixins for DRYer classes.

All mixins in this sub-package define an empty ``__slots__`` slots attribute.

Why do we define ``__slots__`` in the first place?
In short, ``__slots__`` are a more memory efficient way to define attributes
and prevent the creation of the ``__dict__`` and ``__weakref__`` attributes
thus preventing dynamic attribute assignment.

Why do we define an empty ``__slots__`` attribute?
In order to allow for multiple inheritance from these mixins, the mixins
themselves have to not define any slots otherwise Python does not know how to
create such a class, it will raise::

    TypeError: Error when calling the metaclass bases multiple bases have
        instance lay-out conflict

This way of defining the mixin-classes was largely inspired by a very elaborate
answer on StackOverflow [1]_.

References
----------
.. [1] https://stackoverflow.com/a/28059785

"""

from __future__ import absolute_import

from optlang.interface.mixins.subject_mixin import SubjectMixin
from optlang.interface.mixins.solver_state_mixin import SolverStateMixin
from optlang.interface.mixins.name_mixin import NameMixin
from optlang.interface.mixins.bounds_mixin import BoundsMixin
from optlang.interface.mixins.value_mixin import ValueMixin
from optlang.interface.mixins.symbolic_mixin import SymbolicMixin

__all__ = (
    "SubjectMixin", "SolverStateMixin", "NameMixin", "BoundsMixin",
    "ValueMixin", "SymbolicMixin")
