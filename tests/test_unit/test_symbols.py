# -*- coding: utf-8 -*-

# Copyright 2017 Novo Nordisk Foundation Center for Biosustainability,
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

from __future__ import absolute_import

import pytest

import optlang.symbols as symbols


def test_number_from_base():
    assert isinstance(symbols.Number(0), symbols.Basic)


@pytest.mark.parametrize("instance", [
    symbols.Zero,
    symbols.One,
    symbols.NegativeOne,
    symbols.Integer(-8),
    symbols.Integer(8)
])
def test_integer_is_number(instance):
    assert isinstance(instance, symbols.Number)


@pytest.mark.parametrize("number", range(-10, 10))
def test_integer_numeric_equivalence(number):
    assert symbols.Integer(number) == number


def test_real_from_base():
    assert isinstance(symbols.Real(0), symbols.Basic)


@pytest.mark.parametrize("instance", [
    symbols.Real(-8.5),
    symbols.Integer(8.5)
])
def test_real_is_number(instance):
    assert isinstance(instance, symbols.Number)


@pytest.mark.parametrize("number", range(-10, 10))
def test_real_numeric_equivalence(number, abs_tol=1E-15):
    number -= 0.5
    assert abs(symbols.Real(number) - number) < abs_tol


def test_add_identity():
    assert symbols.add() == 0


def test_add_nested():
    args = tuple(range(100))
    assert symbols.add(args) == 4950


@pytest.mark.parametrize("a, b, c", [
    (1, 2, 3),
    (4, 5, 9),
    (1, -2, -1),
    (5, -9, -4)
])
def test_add_pair(a, b, c):
    assert symbols.add(a, b) == c


def test_mul_identity():
    assert symbols.mul() == 1


def test_mul_nested():
    args = tuple(range(1, 10))
    assert symbols.mul(args) == 362880


@pytest.mark.parametrize("a, b, c", [
    (1, 2, 2),
    (4, 5, 20),
    (1, -2, -2),
    (-5, -4, 20)
])
def test_mul_pair(a, b, c):
    assert symbols.mul(a, b) == c


def test_symbol_uniqueness():
    """Expect two symbols to be different objects despite the same name."""
    a = symbols.UniqueSymbol("theone")
    b = symbols.UniqueSymbol("theone")
    assert a.name == b.name
    assert a is not b
