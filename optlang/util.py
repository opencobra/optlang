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


"""Utility functions for optlang."""

import logging

import os

log = logging.getLogger(__name__)
import tempfile
import inspect
from subprocess import check_output
from sympy.printing.str import StrPrinter
import sympy

class Proxy(object):
    __slots__ = ["_obj", "__weakref__"]
    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    #
    # proxying (special cases)
    #
    def __getattribute__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)
    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)
    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_obj"))
    def __str__(self):
        return str(object.__getattribute__(self, "_obj"))
    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))

    #
    # factories
    #
    _special_names = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__',
        '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__',
        '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__',
        '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
        '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__',
        '__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__',
        '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__',
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
        '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__',
        '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__',
        '__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__',
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__',
        '__truediv__', '__xor__', 'next',
    ]

    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""

        def make_method(name):
            def method(self, *args, **kw):
                return getattr(object.__getattribute__(self, "_obj"), name)(*args, **kw)
            return method

        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name):
                namespace[name] = make_method(name)
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)

    def __new__(cls, obj, *args, **kwargs):
        """
        creates an proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an
        __init__ method of their own.
        note: _class_proxy_cache is unique per deriving class (each deriving
        class must hold its own cache)
        """
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[obj.__class__]
        except KeyError:
            cache[obj.__class__] = theclass = cls._create_class_proxy(obj.__class__)
        ins = object.__new__(theclass)
        theclass.__init__(ins, obj, *args, **kwargs)
        return ins

def solve_with_glpsol(glp_prob):
    """Solve glpk problem with glpsol commandline solver. Mainly for testing purposes.

    # Examples
    # --------

    # >>> problem = glp_create_prob()
    # ... glp_read_lp(problem, None, "../tests/data/model.lp")
    # ... solution = solve_with_glpsol(problem)
    # ... print 'asdf'
    # 'asdf'
    # >>> print solution
    # 0.839784

    # Returns
    # -------
    # dict
    #     A dictionary containing the objective value (key ='objval')
    #     and variable primals.
    """
    from swiglpk import glp_get_row_name, glp_get_col_name, glp_write_lp, glp_get_num_rows, glp_get_num_cols

    row_ids = [glp_get_row_name(glp_prob, i) for i in range(1, glp_get_num_rows(glp_prob) + 1)]

    col_ids = [glp_get_col_name(glp_prob, i) for i in range(1, glp_get_num_cols(glp_prob) + 1)]

    with tempfile.NamedTemporaryFile(suffix=".lp", delete=True) as tmp_file:
        tmp_file_name = tmp_file.name
        glp_write_lp(glp_prob, None, tmp_file_name)
        cmd = ['glpsol', '--lp', tmp_file_name, '-w', tmp_file_name + '.sol', '--log', '/dev/null']
        term = check_output(cmd)
        log.info(term)

    try:
        with open(tmp_file_name + '.sol') as sol_handle:
            # print sol_handle.read()
            solution = dict()
            for i, line in enumerate(sol_handle.readlines()):
                if i <= 1 or line == '\n':
                    pass
                elif i <= len(row_ids):
                    solution[row_ids[i - 2]] = line.strip().split(' ')
                elif i <= len(row_ids) + len(col_ids) + 1:
                    solution[col_ids[i - 2 - len(row_ids)]] = line.strip().split(' ')
                else:
                    print(i)
                    print(line)
                    raise Exception("Argggh!")
    finally:
        os.remove(tmp_file_name + ".sol")
    return solution


def glpk_read_cplex(path):
    """Reads cplex file and returns glpk problem.

    Returns
    -------
    glp_prob
        A glpk problems (same type as returned by glp_create_prob)
    """
    from swiglpk import glp_create_prob, glp_read_lp

    problem = glp_create_prob()
    glp_read_lp(problem, None, path)
    return problem


# noinspection PyBroadException
def list_available_solvers():
    """Determine available solver interfaces (with python bindings).

    Returns
    -------
    dict
        A dict like {'GLPK': True, 'GUROBI': False, ...}
    """
    solvers = dict(GUROBI=False, GLPK=False, MOSEK=False, CPLEX=False, SCIPY=False)
    try:
        import gurobipy

        solvers['GUROBI'] = True
        log.debug('Gurobi python bindings found at %s' % os.path.dirname(gurobipy.__file__))
    except Exception:
        log.debug('Gurobi python bindings not available.')
    try:
        import swiglpk

        solvers['GLPK'] = True
        log.debug('GLPK python bindings found at %s' % os.path.dirname(swiglpk.__file__))
    except Exception:
        log.debug('GLPK python bindings not available.')
    try:
        import mosek

        solvers['MOSEK'] = True
        log.debug('Mosek python bindings found at %s' % os.path.dirname(mosek.__file__))
    except Exception:
        log.debug('Mosek python bindings not available.')
    try:
        import cplex

        solvers['CPLEX'] = True
        log.debug('CPLEX python bindings found at %s' % os.path.dirname(cplex.__file__))
    except Exception:
        log.debug('CPLEX python bindings not available.')
    try:
        from scipy import optimize
        optimize.linprog

        solvers["SCIPY"] = True
        log.debug("Scipy linprog function found at %s" % optimize.__file__)
    except (ImportError, AttributeError):
        log.debug("Scipy solver not available")
    return solvers


def inheritdocstring(name, bases, attrs):
    """
    Use as metaclass to inherit class and method docstrings from parent.
    Adapted from http://stackoverflow.com/questions/13937500/inherit-a-parent-class-docstring-as-doc-attribute
    Use this on classes defined in solver-specific interfaces to inherit docstrings from the high-level interface.
    """
    if '__doc__' not in attrs or not attrs["__doc__"]:
        # create a temporary 'parent' to (greatly) simplify the MRO search
        temp = type('temporaryclass', bases, {})
        for cls in inspect.getmro(temp):
            if cls.__doc__ is not None:
                attrs['__doc__'] = cls.__doc__
                break

    for attr_name, attr in attrs.items():
        if not attr.__doc__:
            for cls in inspect.getmro(temp):
                try:
                    if getattr(cls, attr_name).__doc__ is not None:
                        attr.__doc__ = getattr(cls, attr_name).__doc__
                        break
                except (AttributeError, TypeError):
                    continue

    return type(name, bases, attrs)


def method_inheritdocstring(mthd):
    """Use as decorator on a method to inherit doc from parent method of same name"""
    if not mthd.__doc__:
        pass


def is_numeric(obj):
    if isinstance(obj, (int, float)) or getattr(obj, "is_Number", False):
        return True
    else:
        try:
            float(obj)
        except ValueError:
            return False
        else:
            return True


def expr_to_json(expr):
    """
    Converts a Sympy expression to a json-compatible tree-structure.
    """
    if isinstance(expr, sympy.Mul):
        return {"type": "Mul", "args": [expr_to_json(arg) for arg in expr.args]}
    elif isinstance(expr, sympy.Add):
        return {"type": "Add", "args": [expr_to_json(arg) for arg in expr.args]}
    elif isinstance(expr, sympy.Symbol):
        return {"type": "Symbol", "name": expr.name}
    elif isinstance(expr, sympy.Pow):
        return {"type": "Pow", "args": [expr_to_json(arg) for arg in expr.args]}
    elif isinstance(expr, (float, int)):
        return {"type": "Number", "value": expr}
    elif isinstance(expr, sympy.Float):
        return {"type": "Number", "value": float(expr)}
    elif isinstance(expr, sympy.Integer):
        return {"type": "Number", "value": int(expr)}
    else:
        raise NotImplementedError("Type not implemented: " + str(type(expr)))


def parse_expr(expr, local_dict=None):
    """
    Parses a json-object created with 'expr_to_json' into a Sympy expression.

    If a local_dict argument is passed, symbols with be looked up by name, and a new symbol will
    be created only if the name is not in local_dict.
    """
    if local_dict is None:
        local_dict = {}
    if expr["type"] == "Add":
        return sympy.Add._from_args([parse_expr(arg, local_dict) for arg in expr["args"]])
    elif expr["type"] == "Mul":
        return sympy.Mul._from_args([parse_expr(arg, local_dict) for arg in expr["args"]])
    elif expr["type"] == "Pow":
        return sympy.Pow(parse_expr(arg, local_dict) for arg in expr["args"])
    elif expr["type"] == "Symbol":
        try:
            return local_dict[expr["name"]]
        except KeyError:
            return sympy.Symbol(expr["name"])
    elif expr["type"] == "Number":
        return sympy.sympify(expr["value"])
    else:
        raise NotImplementedError(expr["type"] + " is not implemented")


class TemporaryFilename(object):
    """
    Use context manager to create a temporary file that can be opened and closed, and will be deleted in the end.

    Parameters
    ----------
    suffix : str
        The file ending. Default is 'tmp'
    content : str or None
        If str, the content will be written to the file upon creation

    Example
    ----------
    >>> with TemporaryFilename() as tmp_file_name:
    >>>     with open(tmp_file_name, "w") as tmp_file:
    >>>         tmp_file.write(stuff)
    >>>     with open(tmp_file) as tmp_file:
    >>>         stuff = tmp_file.read()
    """
    def __init__(self, suffix="tmp", content=None):
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
        if content is not None:
            tmp_file.write(content)
        self.name = tmp_file.name
        tmp_file.close()

    def __enter__(self):
        return self.name

    def __exit__(self, type, value, traceback):
        os.remove(self.name)


if __name__ == '__main__':
    from swiglpk import glp_create_prob, glp_read_lp, glp_get_num_rows

    problem = glp_create_prob()
    glp_read_lp(problem, None, "../tests/data/model.lp")
    print("asdf", glp_get_num_rows(problem))
    solution = solve_with_glpsol(problem)
    print(solution['R_Biomass_Ecoli_core_w_GAM'])
