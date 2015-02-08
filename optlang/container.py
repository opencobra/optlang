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


class Container(object):
    '''A container for objects that have a name attribute.'''

    def __init__(self, iterable=[]):
        try:
            self._dict = dict(((elem.name, elem) for elem in iterable))
        except AttributeError as e:
            print("Only objects with containing a 'name' attribute can be stored in a Container.")
            raise e
        self._object_list = list(iterable)

    @property
    def _name_list(self):
        return [elem.name for elem in self._object_list]

    @staticmethod
    def _check_for_name_attribute(value):
        if not hasattr(value, 'name'):
            raise AttributeError('Object %s does not have a "name" attribute and cannot not be stored.' % value)

    def _reindex(self):
        self._dict = dict(((elem.name, elem) for elem in self._object_list))

    def __len__(self):
        return len(self._object_list)

    def __contains__(self, key):
        if key in self._dict:
            return True
        elif key in self._object_list:
            return True
        else:
            self._reindex()
            return key in self._dict

    def __iter__(self):
        original_length = len(self._object_list)
        for item in self._object_list.__iter__():
            if original_length != len(self._object_list):
                raise RuntimeError("container changed size during iteration")
            yield item

    def __getitem__(self, key):
        try:
            return self._object_list.__getitem__(key)
        except TypeError:
            try:
                return self._dict[key]
            except KeyError:
                self._reindex()
                try:
                    return self._dict[key]
                except KeyError:
                    raise KeyError("%s does not contain an object with name %s" % (self, key))

    def __setitem__(self, key, value):
        try:
            self._check_for_name_attribute(value)
            self._object_list.__setitem__(key, value)
            self._dict[value.name] = value
        except TypeError:
            try:
                item = self._dict.__getitem__(key)
                index = self._name_list.index(item.name)
                self._dict[key] = value
            except KeyError:
                self._reindex()
                try:
                    self._dict[key] = value
                except KeyError:
                    raise KeyError("%s does not contain an object with name %s" % (self, key))
            self._object_list.__setitem__(index, value)

    def __delitem__(self, key):
        try:
            item = self._object_list.__getitem__(key)
            self._object_list.__delitem__(key)
            self._dict.__delitem__(item.name)
        except TypeError:
            try:
                item = self._dict.__getitem__(key)
                index = self._name_list.index(item.name)
                self._dict.__delitem__(key)
            except KeyError:
                self._reindex()
                try:
                    self._dict.__delitem__(key)
                except KeyError:
                    raise KeyError("%s does not contain an object with name %s" % (self, key))
            self._object_list.__delitem__(index)

    def iterkeys(self):
        return self._name_list.__iter__()

    def keys(self):
        return list(self.iterkeys())

    def itervalues(self):
        return self._object_list.__iter__()

    def values(self):
        return self._object_list

    def iteritems(self):
        for elem in self._object_list:
            yield elem.name, elem

    def fromkeys(self, keys):
        return self.__class__([self.__getitem__(key) for key in keys])

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except (KeyError, IndexError) as e:
            self._reindex()
            try:
                return self.__getitem__(key)
            except (KeyError, IndexError) as e:
                return default

    def clear(self):
        self._object_list = list()
        self._dict = dict()

    def has_key(self, key):
        if key in self._dict:
            return True

    def append(self, value):
        self._check_for_name_attribute(value)
        name = value.name
        if name in self._dict:
            raise Exception("Container '%s' already contains an object with name '%s'." % (self, value.name))
        self._object_list.append(value)
        self._dict[value.name] = value

    def extend(self, values):
        for value in values:
            self._check_for_name_attribute(value)
            if value.name in self._dict:
                raise Exception("Container '%s' already contains an object with name '%s'." % (self, value.name))
        self._object_list.extend(values)
        self._dict.update(dict([(value.name, value) for value in values]))

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError("'%s' object has no attribute %s" % (self, name))

    def __getstate__(self):
        return self._object_list

    def __setstate__(self, obj_list):
        self.__init__(obj_list)

    def __dir__(self):
        attributes = list(self.__class__.__dict__.keys())
        attributes.extend(self._name_list)
        return attributes