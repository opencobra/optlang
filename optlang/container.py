# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
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

from itertools import islice

from optlang.exceptions import ContainerAlreadyContains


class Container(object):
    '''A container for objects that have a name attribute.'''

    def __init__(self, iterable=()):
        self._dict = {}
        self._object_list = []
        self._indices = {}
        for item in iterable:
            try:
                item.name
            except AttributeError:
                raise AttributeError("Item " + item.__repr__() + " does not have a 'name' attribute")
            self.append(item)

    @staticmethod
    def _check_for_name_attribute(value):
        if not hasattr(value, 'name'):
            raise AttributeError('Object %s does not have a "name" attribute and cannot not be stored.' % value)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, item):
        if item in self._dict:
            return True
        elif hasattr(item, "name") and item in self._object_list:
            return True
        return False

    def __iter__(self):
        original_length = len(self._object_list)
        for item in self._object_list.__iter__():
            if original_length != len(self._object_list):
                raise RuntimeError("container changed size during iteration")
            yield item

    def __getitem__(self, item):
        try:
            return self._object_list[item]  # Try treating item as int or slice
        except TypeError:  # Not int or slice
            return self._dict[item]  # Treat item as key (name)

    def __setitem__(self, key, value):
        self._check_for_name_attribute(value)

        if isinstance(key, int):
            old_value = self._object_list[key]
            if old_value.name == value.name:
                self._object_list[key] = value
                self._dict[value.name] = value
            else:
                if value.name in self:
                    raise ValueError("The container already contains an object with the name " + repr(value.name))
                else:
                    self._object_list[key] = value
                    del self._dict[old_value.name]
                    self._dict[value.name] = value
        else:
            if value.name != key:
                raise ValueError("Name of item does not match key")
            try:
                old_value = self._dict[key]
            except KeyError:
                self.append(value)
            else:
                self._dict[old_value.name] = value
                self._object_list[self._indices[old_value.name]] = value

    def __delitem__(self, key):
        name = self[key].name
        index = self._indices[name]
        del self._dict[name]
        del self._object_list[index]
        del self._indices[name]
        self._reindex(index)

    def _reindex(self, start=0):
        for i, item in enumerate(islice(self._object_list, start, len(self))):
            self._indices[item.name] = start + i

    def update_key(self, key):
        item = self._dict[key]
        name = item.name
        if key != name:
            self._dict[name] = item
            del self._dict[key]
            self._indices[name] = self._indices[key]
            del self._indices[key]

    def keys(self):
        return list(item.name for item in self._object_list)

    def iterkeys(self):
        return (item.name for item in self._object_list)

    def values(self):
        return list(iter(self._object_list))

    def itervalues(self):
        return iter(self._object_list)

    def items(self):
        return ((item.name, item) for item in self._object_list)

    def iteritems(self):
        return self.items()

    def fromkeys(self, keys):
        return self.__class__((self[key] for key in keys))

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def clear(self):
        self._dict = {}
        self._indices = {}
        self._object_list = []

    def has_key(self, key):
        return key in self._dict

    def append(self, value):
        self._check_for_name_attribute(value)
        name = value.name
        if name in self._dict:
            raise ContainerAlreadyContains("Container '%s' already contains an object with name '%s'." % (self, value.name))
        self._indices[name] = len(self)
        self._object_list.append(value)
        self._dict[name] = value

    def extend(self, values):
        for value in values:
            self._check_for_name_attribute(value)
            if value.name in self._dict:
                raise ContainerAlreadyContains("Container '%s' already contains an object with name '%s'." % (self, value.name))
        length = len(self)
        self._object_list.extend(values)
        self._dict.update({value.name: value for value in values})
        self._indices.update({value.name: length + i for i, value in enumerate(values)})

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError("'%s' object has no attribute %s" % (self, name))

    def __getstate__(self):
        return self._object_list

    def __setstate__(self, obj_list):
        self.__init__(obj_list)

    def __dir__(self):
        attributes = list(self.__class__.__dict__.keys())
        attributes.extend(item.name for item in self._object_list)
        return attributes
