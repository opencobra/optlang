# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Interface for the GNU Linear Programming Kit (GLPK)

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""

import logging
import types

log = logging.getLogger(__name__)
import tempfile
import sympy
from sympy.core.add import _unevaluated_Add
from sympy.core.mul import _unevaluated_Mul

from glpk.glpkpi import glp_find_col, glp_get_col_prim, glp_get_col_dual, GLP_CV, GLP_IV, GLP_BV, GLP_UNDEF, GLP_FEAS, \
    GLP_INFEAS, GLP_NOFEAS, GLP_OPT, GLP_UNBND, \
    glp_set_col_kind, glp_find_row, glp_get_row_prim, glp_get_row_dual, glp_get_obj_val, glp_set_obj_dir, glp_init_smcp, \
    glp_init_iocp, GLP_MIN, GLP_MAX, glp_iocp, glp_smcp, GLP_ON, GLP_OFF, GLP_MSG_OFF, GLP_MSG_ERR, GLP_MSG_ON, \
    GLP_MSG_ALL, glp_term_out, glp_create_index, glp_create_prob, glp_get_num_rows, glp_get_num_cols, glp_get_col_name, \
    glp_get_col_lb, glp_get_col_ub, glp_get_col_kind, glp_set_prob_name, glp_prob, glp_read_prob, glp_copy_prob, \
    glp_set_obj_coef, glp_simplex, LPX_LP, _glp_lpx_get_class, glp_intopt, glp_get_status, glp_add_cols, \
    glp_set_col_name, intArray, glp_del_cols, glp_add_rows, glp_set_row_name, doubleArray, glp_write_lp, glp_write_prob, \
    glp_set_mat_row, glp_set_col_bnds, glp_set_row_bnds, GLP_FR, GLP_UP, GLP_LO, GLP_FX, GLP_DB, glp_del_rows, \
    glp_get_mat_row, glp_get_row_ub, \
    glp_get_row_type, glp_get_row_lb, glp_get_row_name, glp_get_obj_coef, glp_get_obj_dir, glp_scale_prob, GLP_SF_AUTO

import interface


_GLPK_STATUS_TO_STATUS = {
    GLP_UNDEF: interface.UNDEFINED,
    GLP_FEAS: interface.FEASIBLE,
    GLP_INFEAS: interface.INFEASIBLE,
    GLP_NOFEAS: interface.INFEASIBLE,
    GLP_OPT: interface.OPTIMAL,
    GLP_UNBND: interface.UNBOUNDED
}

_GLPK_VTYPE_TO_VTYPE = {
    GLP_CV: 'continuous',
    GLP_IV: 'integer',
    GLP_BV: 'binary'
}

_VTYPE_TO_GLPK_VTYPE = dict(
    [(val, key) for key, val in _GLPK_VTYPE_TO_VTYPE.iteritems()]
)


class Variable(interface.Variable):
    """..."""

    def __init__(self, name, index=None, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @property
    def index(self):
        try:
            i = glp_find_col(self.problem.problem, self.name)
            if i != 0:
                return i
            else:
                raise Exception(
                    "Could not determine row index for variable %s" % self)
        except:
            return None

    @property
    def primal(self):
        return glp_get_col_prim(self.problem.problem, self.index)

    @property
    def dual(self):
        return glp_get_col_dual(self.problem.problem, self.index)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):

            if name == 'lb' or name == 'ub':
                super(Variable, self).__setattr__(name, value)
                self.problem._glpk_set_col_bounds(self)

            elif name == 'type':
                try:
                    glpk_kind = _VTYPE_TO_GLPK_VTYPE[value]  # FIXME: should this be [value]??
                except KeyError:
                    raise Exception("GLPK cannot handle variables of type %s. \
                        The following variable types are available:\n" +
                                    " ".join(_VTYPE_TO_GLPK_VTYPE.keys()))
                glp_set_col_kind(self.problem.problem, self.index, glpk_kind)
                super(Variable, self).__setattr__(name, value)

            else:
                super(Variable, self).__setattr__(name, value)

        else:
            super(Variable, self).__setattr__(name, value)


class Constraint(interface.Constraint):
    """GLPK solver interface"""

    def __init__(self, expression, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)

    @property
    def index(self):
        try:
            i = glp_find_row(self.problem.problem, self.name)
            if i != 0:
                return i
            else:
                raise Exception(
                    "Could not determine row index for variable %s" % self)
        except:
            return None

    @property
    def primal(self):
        return glp_get_row_prim(self.problem.problem, self.index)

    @property
    def dual(self):
        return glp_get_row_dual(self.problem.problem, self.index)

    def __setattr__(self, name, value):

        super(Constraint, self).__setattr__(name, value)
        if getattr(self, 'problem', None):

            if name == 'name':

                self.problem._glpk_set_row_name(self)

            elif name == 'lb' or name == 'ub':
                self.problem._glpk_set_row_bounds(self)

    def __iadd__(self, other):
        super(Constraint, self).__iadd__(other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            problem_reference._add_constraint(self, sloppy=False)
        return self

    def __isub__(self, other):
        super(Constraint, self).__isub__(other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            problem_reference._add_constraint(self, sloppy=True)
        return self

    def __imul__(self, other):
        super(Constraint, self).__imul__(other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            problem_reference._add_constraint(self, sloppy=True)
        return self

    def __idiv__(self, other):
        super(Constraint, self).__idiv__(other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            problem_reference._add_constraint(self, sloppy=True)
        return self


class Objective(interface.Objective):
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)

    @property
    def value(self):
        return glp_get_obj_val(self.problem.problem)

    @property
    def value(self):
        return glp_get_obj_val(self.problem.problem)

    def __setattr__(self, name, value):

        if getattr(self, 'problem', None):
            if name == 'direction':
                glp_set_obj_dir(self.problem.problem,
                                {'min': GLP_MIN, 'max': GLP_MAX}[value])
            super(Objective, self).__setattr__(name, value)
        else:
            super(Objective, self).__setattr__(name, value)

    def __iadd__(self, other):
        super(Objective, self).__iadd__(other)
        if self.problem is not None:
            self.problem.objective = self
        return self

    def __isub__(self, other):
        super(Objective, self).__isub__(other)
        if self.problem is not None:
            self.problem.objective = self
        return self

    def __imul__(self, other):
        super(Objective, self).__imul__(other)
        if self.problem is not None:
            self.problem.objective = self
        return self

    def __idiv__(self, other):
        super(Objective, self).__idiv__(other)
        if self.problem is not None:
            self.problem.objective = self
        return self


class Configuration(interface.MathematicalProgrammingConfiguration):
    """docstring for Configuration"""

    def __init__(self, presolve=False, verbosity=0, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self._smcp = glp_smcp()
        self._iocp = glp_iocp()
        glp_init_smcp(self._smcp)
        glp_init_iocp(self._iocp)
        self._set_presolve(presolve)
        self._presolve = presolve
        self._set_verbosity(verbosity)
        self._verbosity = verbosity

    def __getstate__(self):
        return {'presolve': self.presolve, 'verbosity': self.verbosity}

    def __setstate__(self, state):
        self.__init__()
        for key, val in state.iteritems():
            setattr(self, key, val)

    def _set_presolve(self, value):
        self._smcp.presolve = {False: GLP_OFF, True: GLP_ON}[value]
        self._iocp.presolve = {False: GLP_OFF, True: GLP_ON}[value]

    def _set_verbosity(self, value):
        if value == 0:
            glp_term_out(GLP_OFF)
            self._smcp.msg_lev = GLP_MSG_OFF
            self._iocp.msg_lev = GLP_MSG_OFF
        elif value == 1:
            glp_term_out(GLP_OFF)
            self._smcp.msg_lev = GLP_MSG_ERR
            self._iocp.msg_lev = GLP_MSG_ERR
        elif value == 2:
            glp_term_out(GLP_OFF)
            self._smcp.msg_lev = GLP_MSG_ON
            self._iocp.msg_lev = GLP_MSG_ON
        elif value == 3:
            glp_term_out(GLP_ON)
            self._smcp.msg_lev = GLP_MSG_ALL
            self._iocp.msg_lev = GLP_MSG_ALL
        else:
            raise Exception(
                "%s is not a valid verbosity level ranging between 0 and 3."
                % value
            )

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        self._set_presolve(value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._set_verbosity(value)
        self._verbosity = value


class Model(interface.Model):
    """GLPK solver interface"""

    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        self.configuration = Configuration()

        if problem is None:
            self.problem = glp_create_prob()
            glp_create_index(self.problem)
            if self.name is not None:
                glp_set_prob_name(self.problem, self.name)

        elif isinstance(problem, glp_prob):
            self.problem = problem
            glp_create_index(self.problem)
            row_num = glp_get_num_rows(self.problem)
            col_num = glp_get_num_cols(self.problem)
            for i in xrange(1, col_num + 1):
                var = Variable(
                    glp_get_col_name(self.problem, i),
                    lb=glp_get_col_lb(self.problem, i),
                    ub=glp_get_col_ub(self.problem, i),
                    problem=self,
                    type=_GLPK_VTYPE_TO_VTYPE[
                        glp_get_col_kind(self.problem, i)]
                )
                # This avoids adding the variable to the glpk problem
                super(Model, self)._add_variable(var)
            var = self.variables.values()

            for j in xrange(1, row_num + 1):
                ia = intArray(col_num + 1)
                da = doubleArray(col_num + 1)
                nnz = glp_get_mat_row(self.problem, j, ia, da)
                lhs = _unevaluated_Add(*[da[i] * var[ia[i] - 1]
                                         for i in range(1, nnz + 1)])
                glpk_row_type = glp_get_row_type(self.problem, j)
                if glpk_row_type == GLP_FX:
                    row_lb = glp_get_row_lb(self.problem, j)
                    row_ub = row_lb
                elif glpk_row_type == GLP_LO:
                    row_lb = glp_get_row_lb(self.problem, j)
                    row_ub = None
                elif glpk_row_type == GLP_UP:
                    row_lb = None
                    row_ub = glp_get_row_ub(self.problem, j)
                elif glpk_row_type == GLP_DB:
                    row_lb = glp_get_row_lb(self.problem, j)
                    row_ub = glp_get_row_ub(self.problem, j)
                elif glpk_row_type == GLP_FR:
                    row_lb = None
                    row_ub = None
                else:
                    raise Exception(
                        "Currently, optlang does not support glpk row type %s"
                        % str(glpk_row_type)
                    )
                    log.exception()
                if isinstance(lhs, int):
                    lhs = sympy.Integer(lhs)
                elif isinstance(lhs, float):
                    lhs = sympy.Real(lhs)
                super(Model, self)._add_constraint(
                    Constraint(lhs, lb=row_lb, ub=row_ub,
                               name=glp_get_row_name(
                                   self.problem, j), problem=self),
                    sloppy=True
                )

            term_generator = (
                (glp_get_obj_coef(self.problem, index), var[index - 1])
                for index in xrange(1, glp_get_num_cols(problem) + 1)
            )
            self._objective = Objective(
                _unevaluated_Add(
                    *[_unevaluated_Mul(sympy.Real(term[0]), term[1]) for term in term_generator if term[0] != 0.]),
                problem=self,
                direction={GLP_MIN: 'min', GLP_MAX:
                    'max'}[glp_get_obj_dir(self.problem)]
            )
        else:
            raise Exception("Provided problem is not a valid GLPK model.")
        glp_scale_prob(self.problem, GLP_SF_AUTO)

    def __getstate__(self):
        glpk_repr = self.__repr__()
        repr_dict = {'glpk_repr': glpk_repr}
        return repr_dict

    def __setstate__(self, repr_dict):
        tmp_file = tempfile.mktemp(suffix=".lp")
        open(tmp_file, 'w').write(repr_dict['glpk_repr'])
        problem = glp_create_prob()
        # print glp_get_obj_coef(problem, 0)
        glp_read_prob(problem, 0, tmp_file)
        self.__init__(problem=problem)

    # For some reason this is slower then the two methods above ...
    # def __getstate__(self):
    #     state = dict()
    #     for key, val in self.__dict__.iteritems():
    #         if key is not 'problem':
    #             state[key] = val
    #     state['glpk_repr'] = self.__repr__()
    #     return state

    # def __setstate__(self, state):
    #     tmp_file = tempfile.mktemp(suffix=".lp")
    #     open(tmp_file, 'w').write(state.pop('glpk_repr'))
    #     problem = glp_create_prob()
    #     glp_read_prob(problem, 0, tmp_file)
    #     glp_create_index(problem)
    #     state['problem'] = problem
    #     self.__init__()
    #     self.__dict__ = state
    #     for var in self.variables.values():
    #         var.problem = self
    #     for constr in self.constraints.values():
    #         constr.problem = self

    def __deepcopy__(self, memo):
        copy_problem = glp_create_prob()
        glp_copy_prob(copy_problem, self.problem, GLP_ON)
        return Model(problem=copy_problem)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):

        if self._objective is not None:
            for i in xrange(1, len(self.variables)+1):
                glp_set_obj_coef(self.problem, i, 0.)
        super(Model, self.__class__).objective.fset(self, value)  # TODO: This needs to be sped up
        # self._objective = value
        expression = self._objective.expression
        if isinstance(expression, types.FloatType) or isinstance(expression, types.IntType):
            pass
        else:
            if expression.is_Atom:
                glp_set_obj_coef(self.problem, expression.index, 1.)
            if expression.is_Mul:
                coeff, var = expression.args
                glp_set_obj_coef(self.problem, var.index, float(coeff))
            elif expression.is_Add:
                for term in expression.args:
                    coeff, var = term.args
                    glp_set_obj_coef(self.problem, var.index, float(coeff))
            else:
                raise ValueError(
                    "Provided objective %s doesn't seem to be appropriate." %
                    self._objective)
            glp_set_obj_dir(
                self.problem,
                {'min': GLP_MIN, 'max': GLP_MAX}[self._objective.direction]
            )
        value.problem = self

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        glp_write_lp(self.problem, None, tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def __repr__(self):
        tmp_file = tempfile.mktemp(suffix=".glpk")
        glp_write_prob(self.problem, 0, tmp_file)
        glpk_form = open(tmp_file).read()
        return glpk_form

    def optimize(self):
        if _glp_lpx_get_class(self.problem) == LPX_LP:
            glp_simplex(self.problem, self.configuration._smcp)
        else:
            glp_intopt(self.problem, self.configuration._iocp)
            if _GLPK_STATUS_TO_STATUS[glp_get_status(self.problem)] == 'undefined':
                original_presolve_setting = self.configuration.presolve
                self.configuration.presolve = True
                glp_intopt(self.problem, self.configuration._iocp)
                self.configuration.presolve = original_presolve_setting
        glpk_status = glp_get_status(self.problem)
        self.status = _GLPK_STATUS_TO_STATUS[glpk_status]
        return self.status

    def _add_variable(self, variable):
        super(Model, self)._add_variable(variable)
        glp_add_cols(self.problem, 1)
        index = glp_get_num_cols(self.problem)
        glp_set_col_name(self.problem, index, variable.name)
        variable.problem = self
        self._glpk_set_col_bounds(variable)
        glp_set_col_kind(self.problem, variable.index, _VTYPE_TO_GLPK_VTYPE[variable.type])
        return variable

    def _remove_variable(self, variable):
        num = intArray(2)
        num[1] = variable.index
        glp_del_cols(self.problem, 1, num)
        super(Model, self)._remove_variable(variable)

    def _add_constraint(self, constraint, sloppy=False):
        if sloppy is False:
            if not constraint.is_Linear:
                raise ValueError(
                    "GLPK only supports linear constraints. %s is not linear." % constraint)
        super(Model, self)._add_constraint(constraint, sloppy=sloppy)
        constraint.problem = self
        glp_add_rows(self.problem, 1)
        index = glp_get_num_rows(self.problem)
        glp_set_row_name(self.problem, index, constraint.name)
        num_vars = len(constraint.variables)
        index_array = intArray(num_vars + 1)
        value_array = doubleArray(num_vars + 1)
        if constraint.expression.is_Atom and constraint.expression.is_Symbol:
            var = constraint.expression
            index_array[1] = var.index
            value_array[1] = 1
        elif constraint.expression.is_Mul:
            args = constraint.expression.args
            if len(args) > 2:
                raise Exception(
                    "Term %s from constraint %s is not a proper linear term." % (term, constraint))
            coeff = float(args[0])
            var = args[1]
            index_array[1] = var.index
            value_array[1] = coeff
        else:
            for i, term in enumerate(constraint.expression.args):
                args = term.args
                if args == ():
                    assert term.is_Symbol
                    coeff = 1
                    var = term
                elif len(args) == 2:
                    assert args[0].is_Number
                    assert args[1].is_Symbol
                    var = args[1]
                    coeff = float(args[0])
                elif len(args) > 2:
                    raise Exception(
                        "Term %s from constraint %s is not a proper linear term." % (term, constraint))
                index_array[i + 1] = var.index
                value_array[i + 1] = coeff
        glp_set_mat_row(self.problem, index, num_vars,
                        index_array, value_array)
        self._glpk_set_row_bounds(constraint)

    def _glpk_set_col_bounds(self, variable):
        if variable.lb is None and variable.ub is None:
            # 0.'s are ignored
            glp_set_col_bnds(self.problem, variable.index, GLP_FR, 0., 0.)
        elif variable.lb is None:
            # 0. is ignored
            glp_set_col_bnds(self.problem, variable.index,
                             GLP_UP, 0., float(variable.ub))
        elif variable.ub is None:
            # 0. is ignored
            glp_set_col_bnds(self.problem, variable.index,
                             GLP_LO, float(variable.lb), 0.)
        elif variable.lb == variable.ub:
            glp_set_col_bnds(self.problem, variable.index,
                             GLP_FX, float(variable.lb), float(variable.lb))
        elif variable.lb < variable.ub:
            glp_set_col_bnds(self.problem, variable.index,
                             GLP_DB, float(variable.lb), float(variable.ub))
        elif variable.lb > variable.ub:
            raise ValueError(
                "Lower bound %f is larger than upper bound %f in variable %s" %
                (variable.lb, variable.ub, variable))
        else:
            raise Exception(
                "Something is wrong with the provided bounds %f and %f in variable %s" %
                (variable.lb, variable.ub, variable))

    def _glpk_set_row_bounds(self, constraint):
        if constraint.lb is None and constraint.ub is None:
            # 0.'s are ignored
            glp_set_row_bnds(self.problem, constraint.index, GLP_FR, 0., 0.)
        elif constraint.lb is None:
            # 0. is ignored
            glp_set_row_bnds(self.problem, constraint.index,
                             GLP_UP, 0., float(constraint.ub))
        elif constraint.ub is None:
            # 0. is ignored
            glp_set_row_bnds(self.problem, constraint.index,
                             GLP_LO, float(constraint.lb), 0.)
        elif constraint.lb is constraint.ub:
            glp_set_row_bnds(self.problem, constraint.index,
                             GLP_FX, float(constraint.lb), float(constraint.lb))
        elif constraint.lb < constraint.ub:
            glp_set_row_bnds(self.problem, constraint.index,
                             GLP_DB, float(constraint.lb), float(constraint.ub))
        elif constraint.lb > constraint.ub:
            raise ValueError(
                "Lower bound %f is larger than upper bound %f in constraint %s" %
                (constraint.lb, constraint.ub, constraint))
        else:
            raise Exception(
                "Something is wrong with the provided bounds %f and %f in constraint %s" %
                (constraint.lb, constraint.ub, constraint))

    def _remove_constraint(self, constraint):
        num = intArray(2)
        num[1] = constraint.index
        glp_del_rows(self.problem, 1, num)
        super(Model, self)._remove_constraint(constraint)


if __name__ == '__main__':
    import pickle

    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)
    c1 = Constraint(x1 + x2 + x3, lb=-100, ub=100, name='c1')
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])
    status = model.optimize()
    print "status:", model.status
    print "objective value:", model.objective.value

    for var_name, var in model.variables.iteritems():
        print var_name, "=", var.primal

    print model

    from glpk.glpkpi import glp_read_lp

    problem = glp_create_prob()
    glp_read_lp(problem, None, "../tests/data/model.lp")

    solver = Model(problem=problem)
    print solver.optimize()
    print solver.objective

    import time

    t1 = time.time()
    print "pickling"
    pickle_string = pickle.dumps(solver)
    resurrected_solver = pickle.loads(pickle_string)
    t2 = time.time()
    print "Execution time: %s" % (t2 - t1)

    resurrected_solver.optimize()
    print "Halelujah!", resurrected_solver.objective.value