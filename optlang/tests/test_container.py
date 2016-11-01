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
import pickle
import unittest

from optlang.container import Container
from optlang.interface import Variable, Model


class ContainerTestCase(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.container = self.model.variables

    def test_container_from_iterable(self):
        variables_iterable = [Variable("v" + str(i), lb=10, ub=100) for i in range(10000)]
        container = Container(variables_iterable)
        self.assertEqual(len(container), 10000)
        for index, variable in enumerate(variables_iterable):
            self.assertEqual(container[index], variable)
            self.assertEqual(container[variable.name], variable)

    def test_container_from_iterable_with_items_without_name_attribute_raise(self):
        variables_iterable = ['v' + str(i) for i in range(10000)]
        self.assertRaises(AttributeError, Container, variables_iterable)

    def test_container_append(self):
        var = Variable('blub')
        self.container.append(var)
        self.assertEqual(len(self.container), 1)
        self.assertEqual(self.container['blub'], var)
        self.assertEqual(self.container[0], var)

    def test_dir(self):
        var = Variable('blub')
        self.container.append(var)
        print(dir(self.container))
        self.assertEqual(dir(self.container),
                         ['__contains__', '__delitem__', '__dict__', '__dir__', '__doc__', '__getattr__', '__getitem__',
                          '__getstate__', '__init__', '__iter__', '__len__', '__module__', '__setitem__',
                          '__setstate__', '__weakref__', '_check_for_name_attribute', '_reindex', 'append', 'blub',
                          'clear', 'extend', 'fromkeys', 'get', 'has_key', 'items', 'iteritems', 'iterkeys',
                          'itervalues', 'keys', 'update_key', 'values'])

    def test_del_by_index(self):
        variables_iterable = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        container = Container(variables_iterable)
        del container[10]
        for i, variable in enumerate(container):
            if i < 10:
                self.assertEqual(int(variable.name.replace('v', '')), i)
            else:
                self.assertEqual(int(variable.name.replace('v', '')) - 1, i)

    def test_del_by_key(self):
        variables_iterable = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        container = Container(variables_iterable)
        del container["v333"]
        for i, variable in enumerate(container):
            if i < 333:
                self.assertEqual(int(variable.name.replace('v', '')), i)
            else:
                self.assertEqual(int(variable.name.replace('v', '')) - 1, i)

    def test_non_name_item_raises(self):
        self.assertRaises(AttributeError, self.container.append, 3)

    def test_add_already_existing_item_raises(self):
        var = Variable('blub')
        self.container.append(var)
        self.assertRaises(Exception, self.container.append, var)

        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(100)]
        self.container.extend(variables)
        self.assertRaises(Exception, self.container.extend, variables)

    def test_clear(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        self.container.clear()
        self.assertEqual(len(self.container), 0)
        self.assertEqual(self.container._object_list, [])
        self.assertEqual(self.container._indices, {})
        self.assertEqual(self.container._dict, {})

    def test_extend(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)

    def test_iterkeys(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        generator = self.container.iterkeys()
        self.assertEqual(list(generator), [item.name for item in self.container])

    def test_keys(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        keys = self.container.keys()
        self.assertEqual(keys, [item.name for item in self.container])

    def test_itervalues(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        generator = self.container.itervalues()
        self.assertEqual(list(generator), variables)

    def test_values(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        values = self.container.values()
        self.assertEqual(values, variables)

    def test_iteritems(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        generator = self.container.iteritems()
        self.assertEqual(list(generator), [(variable.name, variable) for variable in variables])

    def test_fromkeys(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(1000)]
        self.container.extend(variables)
        sub_container = self.container.fromkeys(('v1', 'v66', 'v999'))
        print(sub_container._object_list)
        lookalike = Container([variables[i] for i in (1, 66, 999)])
        print(lookalike._object_list)
        self.assertEqual(sub_container._object_list, lookalike._object_list)
        # self.assertEqual(sub_container._name_list, lookalike._name_list)
        self.assertEqual(sub_container._dict, lookalike._dict)

    def test_get(self):
        var = Variable('blub')
        self.container.append(var)
        self.assertEqual(self.container.get('blub'), var)
        self.assertEqual(self.container.get('blurb', None), None)

    def test_has_key(self):
        self.assertFalse('blurb' in self.container)
        self.assertFalse(self.container.has_key('blurb'))  # noqa: W601
        var = Variable('blurb')
        self.container.append(var)
        self.assertTrue('blurb' in self.container)
        self.assertTrue(self.container.has_key('blurb'))  # noqa: W601

    def test_getattr(self):
        var = Variable('variable1')
        self.container.append(var)
        self.assertEqual(self.container.variable1, var)
        self.assertRaises(AttributeError, getattr, self.container, 'not_existing_variable')

    def test_setitem(self):
        var = Variable('blub')
        self.assertRaises(IndexError, self.container.__setitem__, 0, var)
        self.container.append(var)
        self.assertEqual(self.container[0], var)
        var2 = Variable('blib')
        self.container[0] = var2
        self.assertEqual(self.container[0], var2)

        var3 = Variable("blab")
        self.assertRaises(ValueError, self.container.__setitem__, "blub", var3)
        self.container["blab"] = var3
        self.assertIs(self.container["blab"], var3)
        self.assertIs(self.container[1], var3)

        var4 = Variable("blab")
        self.container["blab"] = var4
        self.assertFalse(var3 in self.container)
        self.assertIs(self.container["blab"], var4)
        self.assertIs(self.container[1], var4)

        self.assertRaises(ValueError, self.container.__setitem__, 1, var2)
        self.container[1] = var3
        self.assertIs(self.container["blab"], var3)
        self.assertIs(self.container[1], var3)
        self.container.update_key("blab")

    def test_change_object_name(self):
        var = Variable('blub')
        self.model.add(var)
        self.model.update()
        var.name = 'blurg'
        self.assertEqual(self.container.keys(), ['blurg'])
        self.assertEqual(self.container['blurg'], var)

    def test_iter_container_len_change_raises(self):
        def _(container):
            for item in container:
                del container[item.name]

        variables_iterable = [Variable("v" + str(i), lb=10, ub=100) for i in range(10)]
        container = Container(variables_iterable)
        self.assertRaises(RuntimeError, _, container)

    def test_pickle(self):
        variables = [Variable("v" + str(i), lb=10, ub=100) for i in range(100)]
        self.container.extend(variables)
        unpickled = pickle.loads(pickle.dumps(self.container))
        self.assertEquals(unpickled[0].name, variables[0].name)
