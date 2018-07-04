# Copyright 2016 Novo Nordisk Foundation Center for Biosustainability,
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

from optlang.symbolics import Integer

one = Integer(1)


def parse_optimization_expression(obj, linear=True, quadratic=False, expression=None, **kwargs):
    """
    Function for parsing the expression of a Constraint or Objective object.

    Parameters
    ----------
    object: Constraint or Objective
        The optimization expression to be parsed
    linear: Boolean
        If True the expression will be assumed to be linear
    quadratic: Boolean
        If True the expression will be assumed to be quadratic
    expression: Sympy expression or None (optional)
        An expression can be passed explicitly to avoid getting the expression from the solver.
        If this is used then 'linear' or 'quadratic' should be True.

    If both linear and quadratic are False, the is_Linear and is_Quadratic methods will be used to determine how it should be parsed

    Returns
    ----------
    A tuple of (linear_coefficients, quadratic_coefficients)
        linear_coefficients is a dictionary of {variable: coefficient} pairs
        quadratic_coefficients is a dictionary of {frozenset(variables): coefficient} pairs
    """
    if expression is None:
        expression = obj.expression

    if not (linear or quadratic):
        if obj.is_Linear:
            linear = True
        elif obj.is_Quadratic:
            quadratic = True
        else:
            raise ValueError("Expression is not linear or quadratic. Other expressions are not currently supported.")

    assert linear or quadratic

    if quadratic:
        offset, linear_coefficients, quadratic_coefficients = _parse_quadratic_expression(expression, **kwargs)
    else:
        offset, linear_coefficients = _parse_linear_expression(expression, **kwargs)
        quadratic_coefficients = {}

    return offset, linear_coefficients, quadratic_coefficients


def _parse_linear_expression(expression, expanded=False, **kwargs):
    """
    Parse the coefficients of a linear expression (linearity is assumed).

    Returns a dictionary of variable: coefficient pairs.
    """
    offset = 0
    constant = None

    if expression.is_Add:
        coefficients = expression.as_coefficients_dict()
    elif expression.is_Mul:
        coefficients = {expression.args[1]: expression.args[0]}
    elif expression.is_Symbol:
        coefficients = {expression: 1}
    elif expression.is_Number:
        coefficients = {}
    else:
        raise ValueError("Expression {} seems to be invalid".format(expression))

    for var in coefficients:
        if not (var.is_Symbol):
            if var == one:
                constant = var
                offset = float(coefficients[var])
            elif expanded:
                raise ValueError("Expression {} seems to be invalid".format(expression))
            else:
                coefficients = _parse_linear_expression(expression, expanded=True, **kwargs)
    if constant is not None:
        del coefficients[constant]
    return offset, coefficients


def _parse_quadratic_expression(expression, expanded=False):
    """
    Parse a quadratic expression. It is assumed that the expression is known to be quadratic or linear.

    The 'expanded' parameter tells whether the expression has already been expanded. If it hasn't the parsing
    might fail and will expand the expression and try again.
    """
    linear_coefficients = {}
    quadratic_coefficients = {}
    offset = 0

    if expression.is_Number:  # Constant expression, no coefficients
        return float(expression), linear_coefficients, quadratic_coefficients

    if expression.is_Mul:
        terms = (expression,)
    elif expression.is_Add:
        terms = expression.args
    else:
        raise ValueError("Expression of type {} could not be parsed.".format(type(expression)))

    try:
        for term in terms:
            if term.is_Number:
                offset += float(term)
                continue
            if term.is_Pow:
                term = 1.0 * term
            assert term.is_Mul, "What is this? {}".format(type(term))
            factors = term.args
            coef = factors[0]
            vars = factors[1:]
            assert len(vars) <= 2, "This should not happen. Is this expression quadratic?"
            if len(vars) == 2:
                key = frozenset(vars)
                quadratic_coefficients[key] = quadratic_coefficients.get(key, 0) + coef
            else:
                var = vars[0]
                if var.is_Symbol:
                    linear_coefficients[var] = linear_coefficients.get(var, 0) + coef
                elif var.is_Pow:
                    var, exponent = var.args
                    if exponent != 2:
                        raise ValueError("The expression is not quadratic")
                    key = frozenset((var,))
                    quadratic_coefficients[key] = quadratic_coefficients.get(key, 0) + coef
        if quadratic_coefficients:
            assert all(var.is_Symbol for var in frozenset.union(*quadratic_coefficients))  # Raise an exception to trigger expand
        if linear_coefficients:
            assert all(var.is_Symbol for var in linear_coefficients)
    except Exception as e:
        if expanded:
            raise e
        else:
            # Try to expand the expression and parse it again
            return _parse_quadratic_expression(expression.expand(), expanded=True)

    return offset, linear_coefficients, quadratic_coefficients
