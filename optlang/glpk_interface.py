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


"""Interface for the GNU Linear Programming Kit (GLPK)

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.
"""

import collections
import logging
import tempfile

import six
import sympy
from sympy.core.add import _unevaluated_Add
from sympy.core.mul import _unevaluated_Mul

from optlang.util import inheritdocstring

log = logging.getLogger(__name__)

from swiglpk import glp_find_col, glp_get_col_prim, glp_get_col_dual, GLP_CV, GLP_IV, GLP_BV, GLP_UNDEF, GLP_FEAS, \
    GLP_INFEAS, GLP_NOFEAS, GLP_OPT, GLP_UNBND, \
    glp_set_col_kind, glp_find_row, glp_get_row_prim, glp_get_row_dual, glp_get_obj_val, glp_set_obj_dir, glp_init_smcp, \
    glp_init_iocp, GLP_MIN, GLP_MAX, glp_iocp, glp_smcp, GLP_ON, GLP_OFF, GLP_MSG_OFF, GLP_MSG_ERR, GLP_MSG_ON, \
    GLP_MSG_ALL, glp_term_out, glp_create_index, glp_create_prob, glp_get_num_rows, glp_get_num_cols, glp_get_col_name, \
    glp_get_col_lb, glp_get_col_ub, glp_get_col_kind, glp_set_prob_name, glp_read_prob, glp_set_obj_coef, glp_simplex, \
    glp_intopt, glp_get_status, glp_add_cols, \
    glp_set_col_name, intArray, glp_del_cols, glp_add_rows, glp_set_row_name, doubleArray, glp_write_lp, glp_write_prob, \
    glp_set_mat_row, glp_set_col_bnds, glp_set_row_bnds, GLP_FR, GLP_UP, GLP_LO, GLP_FX, GLP_DB, glp_del_rows, \
    glp_get_mat_row, glp_get_row_ub, glp_get_row_type, glp_get_row_lb, glp_get_row_name, glp_get_obj_coef, \
    glp_get_obj_dir, glp_scale_prob, GLP_SF_AUTO, glp_get_num_int, glp_get_num_bin, glp_mip_col_val, \
    glp_mip_obj_val, glp_mip_status, GLP_ETMLIM

from optlang import interface

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
    [(val, key) for key, val in six.iteritems(_GLPK_VTYPE_TO_VTYPE)]
)


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, index=None, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @property
    def index(self):
        if self.problem is not None:
            i = glp_find_col(self.problem.problem, str(self.name))
            if i != 0:
                return i
            else:
                raise IndexError(
                    "Could not determine column index for variable %s" % self)
        else:
            return None

    @interface.Variable.lb.setter
    def lb(self, value):
        interface.Variable.lb.fset(self, value)
        self.problem._glpk_set_col_bounds(self)

    @interface.Variable.ub.setter
    def ub(self, value):
        interface.Variable.ub.fset(self, value)
        self.problem._glpk_set_col_bounds(self)

    @interface.Variable.type.setter
    def type(self, value):
        try:
            glpk_kind = _VTYPE_TO_GLPK_VTYPE[value]
        except KeyError:
            raise Exception("GLPK cannot handle variables of type %s. \
                        The following variable types are available:\n" +
                            " ".join(_VTYPE_TO_GLPK_VTYPE.keys()))
        glp_set_col_kind(self.problem.problem, self.index, glpk_kind)
        interface.Variable.type.fset(self, value)

    @property
    def primal(self):
        if self.problem:
            if self.type == "continuous":
                primal_from_solver = glp_get_col_prim(self.problem.problem, self.index)
            elif self.type in ["binary", "integer"]:
                primal_from_solver = glp_mip_col_val(self.problem.problem, self.index)
            else:
                raise TypeError("Unknown variable type")
            return self._round_primal_to_bounds(primal_from_solver)
        else:
            return None

    @property
    def dual(self):
        if self.problem:
            return glp_get_col_dual(self.problem.problem, self.index)
        else:
            return None

    @interface.Variable.name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        self._name = value
        if getattr(self, 'problem', None) is not None:
            glp_set_col_name(self.problem.problem, glp_find_col(self.problem.problem, old_name), str(value))
            self.problem.variables.update_key(old_name)


@six.add_metaclass(inheritdocstring)
class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Constraint, self).__init__(expression, sloppy=sloppy, *args, **kwargs)
        if not sloppy:
            if not self.is_Linear:
                raise ValueError(
                    "GLPK only supports linear constraints. %s is not linear." % self)

    def _get_expression(self):
        if self.problem is not None:
            col_num = glp_get_num_cols(self.problem.problem)
            ia = intArray(col_num + 1)
            da = doubleArray(col_num + 1)
            nnz = glp_get_mat_row(self.problem.problem, self.index, ia, da)
            constraint_variables = [self.problem._variables[glp_get_col_name(self.problem.problem, ia[i])] for i in
                                    range(1, nnz + 1)]
            expression = sympy.Add._from_args(
                [sympy.Mul._from_args((sympy.RealNumber(da[i]), constraint_variables[i - 1])) for i in
                 range(1, nnz + 1)])
            self._expression = expression
        return self._expression

    def _set_coefficients_low_level(self, variables_coefficients_dict):
        if self.problem is not None:
            problem = self.problem.problem
            indices_coefficients_dict = dict(
                [(variable.index, coefficient) for variable, coefficient in six.iteritems(variables_coefficients_dict)])
            num_cols = glp_get_num_cols(problem)
            ia = intArray(num_cols + 1)
            da = doubleArray(num_cols + 1)
            index = self.index
            num = glp_get_mat_row(self.problem.problem, index, ia, da)
            for i in range(1, num + 1):
                try:
                    da[i] = indices_coefficients_dict[ia[i]]
                except KeyError:
                    pass
            glp_set_mat_row(self.problem.problem, index, num, ia, da)
        else:
            raise Exception(
                '_set_coefficients_low_level works only if a constraint is associated with a solver instance.')

    @interface.Constraint.lb.setter
    def lb(self, value):
        self._lb = value
        if self.problem is not None:
            self.problem._glpk_set_row_bounds(self)

    @interface.Constraint.ub.setter
    def ub(self, value):
        self._ub = value
        if self.problem is not None:
            self.problem._glpk_set_row_bounds(self)

    @interface.OptimizationExpression.name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        self._name = value
        if self.problem is not None:
            glp_set_row_name(self.problem.problem, glp_find_row(self.problem.problem, old_name), str(value))
            self.problem.constraints.update_key(old_name)

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        if value is None:
            # Update expression from solver instance one last time
            self._get_expression()
            self._problem = None
        else:
            self._problem = value

    @property
    def index(self):
        try:
            i = glp_find_row(self.problem.problem, str(self.name))
            if i != 0:
                return i
            else:
                raise IndexError(
                    "Could not determine row index for variable %s" % self)
        except:
            return None

    @property
    def primal(self):
        if self.problem is not None:
            primal_from_solver = glp_get_row_prim(self.problem.problem, self.index)
            # return self._round_primal_to_bounds(primal_from_solver)  # Test assertions fail
            return primal_from_solver
        else:
            return None

    @property
    def dual(self):
        if self.problem is not None:
            return glp_get_row_dual(self.problem.problem, self.index)
        else:
            return None

    def __iadd__(self, other):
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            super(Constraint, self).__iadd__(other)
            problem_reference._add_constraint(self, sloppy=False)
        else:
            super(Constraint, self).__iadd__(other)
        return self


@six.add_metaclass(inheritdocstring)
class Objective(interface.Objective):
    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)
        if not self.is_Linear:
            raise ValueError(
                "GLPK only supports linear objectives. %s is not linear." % self)

    def _get_expression(self):
        if self.problem is not None:
            variables = self.problem._variables

            def term_generator():
                for index in range(1, glp_get_num_cols(self.problem.problem) + 1):
                    coeff = glp_get_obj_coef(self.problem.problem, index)
                    if coeff != 0.:
                        yield (sympy.RealNumber(coeff), variables[index - 1])

            expression = sympy.Add._from_args([sympy.Mul._from_args(term) for term in term_generator()])
            self._expression = expression
        return self._expression

    @property
    def value(self):
        if (glp_get_num_int(self.problem.problem) + glp_get_num_bin(self.problem.problem)) > 0:
            return glp_mip_obj_val(self.problem.problem)
        else:
            return glp_get_obj_val(self.problem.problem)

    @interface.Objective.direction.setter
    def direction(self, value):
        if getattr(self, 'problem', None) is not None:
            glp_set_obj_dir(self.problem.problem, {'min': GLP_MIN, 'max': GLP_MAX}[value])
        super(Objective, Objective).direction.__set__(self, value)

    def __iadd__(self, other):
        self.problem = None
        super(Objective, self).__iadd__(other)
        if self.problem is not None:
            self.problem.objective = self
        return self

    def __imul__(self, other):
        self.problem = None
        super(Objective, self).__imul__(other)
        if self.problem is not None:
            self.problem.objective = self
        return self


@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, presolve="auto", verbosity=0, timeout=None, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self._smcp = glp_smcp()
        self._iocp = glp_iocp()
        glp_init_smcp(self._smcp)
        glp_init_iocp(self._iocp)
        self._max_time = min(self._smcp.tm_lim, self._iocp.tm_lim)
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout

    def __getstate__(self):
        return {'presolve': self.presolve, 'verbosity': self.verbosity, 'timeout': self.timeout}

    def __setstate__(self, state):
        self.__init__()
        for key, val in six.iteritems(state):
            setattr(self, key, val)

    def _set_presolve(self, value):
        self._smcp.presolve = {False: GLP_OFF, True: GLP_ON, "auto": GLP_OFF}[value]
        self._iocp.presolve = {False: GLP_OFF, True: GLP_ON, "auto": GLP_OFF}[value]

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

    def _set_timeout(self, value):
        if value is None:
            self._smcp.tm_lim = self._max_time
            self._iocp.tm_lim = self._max_time
        else:
            self._smcp.tm_lim = value * 1000  # milliseconds to seconds
            self._iocp.tm_lim = value * 1000

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

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._set_timeout(value)
        self._timeout = value


@six.add_metaclass(inheritdocstring)
class Model(interface.Model):
    def __init__(self, problem=None, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        self.configuration = Configuration()

        if problem is None:
            self.problem = glp_create_prob()
            glp_create_index(self.problem)
            if self.name is not None:
                glp_set_prob_name(self.problem, str(self.name))

        else:
            try:
                self.problem = problem
                glp_create_index(self.problem)
            except TypeError:
                raise TypeError("Provided problem is not a valid GLPK model.")
            row_num = glp_get_num_rows(self.problem)
            col_num = glp_get_num_cols(self.problem)
            for i in range(1, col_num + 1):
                var = Variable(
                    glp_get_col_name(self.problem, i),
                    lb=glp_get_col_lb(self.problem, i),
                    ub=glp_get_col_ub(self.problem, i),
                    problem=self,
                    type=_GLPK_VTYPE_TO_VTYPE[
                        glp_get_col_kind(self.problem, i)]
                )
                # This avoids adding the variable to the glpk problem
                super(Model, self)._add_variables([var])
            variables = self.variables

            for j in range(1, row_num + 1):
                ia = intArray(col_num + 1)
                da = doubleArray(col_num + 1)
                nnz = glp_get_mat_row(self.problem, j, ia, da)
                constraint_variables = [variables[ia[i] - 1] for i in range(1, nnz + 1)]
                lhs = _unevaluated_Add(*[da[i] * constraint_variables[i - 1]
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
                    lhs = sympy.RealNumber(lhs)
                constraint_id = glp_get_row_name(self.problem, j)
                for variable in constraint_variables:
                    try:
                        self._variables_to_constraints_mapping[variable.name].add(constraint_id)
                    except KeyError:
                        self._variables_to_constraints_mapping[variable.name] = set([constraint_id])

                super(Model, self)._add_constraints(
                    [Constraint(lhs, lb=row_lb, ub=row_ub, name=constraint_id, problem=self)], sloppy=True)

            term_generator = (
                (glp_get_obj_coef(self.problem, index), variables[index - 1])
                for index in range(1, glp_get_num_cols(problem) + 1)
            )
            self._objective = Objective(
                _unevaluated_Add(
                    *[_unevaluated_Mul(sympy.RealNumber(term[0]), term[1]) for term in term_generator if
                      term[0] != 0.]),
                problem=self,
                direction={GLP_MIN: 'min', GLP_MAX: 'max'}[glp_get_obj_dir(self.problem)])
        glp_scale_prob(self.problem, GLP_SF_AUTO)

    def __getstate__(self):
        self.update()
        glpk_repr = self._glpk_representation()
        repr_dict = {'glpk_repr': glpk_repr, 'glpk_status': self.status, 'config': self.configuration}
        return repr_dict

    def __setstate__(self, repr_dict):
        tmp_file = tempfile.mktemp(suffix=".glpk")
        open(tmp_file, 'w').write(repr_dict['glpk_repr'])
        problem = glp_create_prob()
        glp_read_prob(problem, 0, tmp_file)
        self.__init__(problem=problem)
        self.configuration = Configuration.clone(repr_dict['config'], problem=self)
        if repr_dict['glpk_status'] == 'optimal':
            self.optimize()  # since the start is an optimal solution, nothing will happen here

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        value.problem = None
        if self._objective is not None:
            for variable in self.objective.variables:
                if variable.index is not None:
                    glp_set_obj_coef(self.problem, variable.index, 0.)
        super(Model, self.__class__).objective.fset(self, value)
        self.update()
        expression = self._objective._expression
        if isinstance(expression, float) or isinstance(expression, int) or expression.is_Number:
            pass
        else:
            if expression.is_Symbol:
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

    @property
    def primal_values(self):
        if self.problem:
            primal_values = collections.OrderedDict()
            for index, variable in enumerate(self.variables):
                if variable.type == "continuous":
                    value = glp_get_col_prim(self.problem, index + 1)
                elif variable.type in ["binary", "integer"]:
                    value = glp_mip_col_val(self.problem, index + 1)
                else:
                    raise TypeError("Unknown variable type")
                primal_values[variable.name] = variable._round_primal_to_bounds(value)
            return primal_values
        else:
            return None

    @property
    def reduced_costs(self):
        if self.problem:
            reduced_costs = collections.OrderedDict()
            for index, variable in enumerate(self.variables):
                if variable.type == "continuous":
                    value = glp_get_col_dual(self.problem, index + 1)
                elif variable.type in ["binary", "integer"]:
                    value = glp_mip_col_val(self.problem, index + 1)
                else:
                    raise TypeError("Unknown variable type")
                reduced_costs[variable.name] = value
            return reduced_costs
        else:
            return None

    @property
    def dual_values(self):
        if self.problem:
            dual_values = collections.OrderedDict()
            for index, constraint in enumerate(self.constraints):
                value = glp_get_row_prim(self.problem, index + 1)
                dual_values[constraint.name] = value
            return dual_values
        else:
            return None

    @property
    def shadow_prices(self):
        if self.problem:
            shadow_prices = collections.OrderedDict()
            for index, constraint in enumerate(self.constraints):
                value = glp_get_row_dual(self.problem, index + 1)
                shadow_prices[constraint.name] = value
            return shadow_prices
        else:
            return None

    def __str__(self):
        tmp_file = tempfile.mktemp(suffix=".lp")
        glp_write_lp(self.problem, None, tmp_file)
        cplex_form = open(tmp_file).read()
        return cplex_form

    def _glpk_representation(self):
        tmp_file = tempfile.mktemp(suffix=".glpk")
        glp_write_prob(self.problem, 0, tmp_file)
        glpk_form = open(tmp_file).read()
        return glpk_form

    def _run_glp_simplex(self):
        return_value = glp_simplex(self.problem, self.configuration._smcp)
        glpk_status = glp_get_status(self.problem)
        if return_value == 0:
            status = _GLPK_STATUS_TO_STATUS[glpk_status]
        elif return_value == GLP_ETMLIM:
            status = interface.TIME_LIMIT
        else:
            status = _GLPK_STATUS_TO_STATUS[glpk_status]
            if status == interface.UNDEFINED:
                log.debug("Status undefined. GLPK status code returned by glp_simplex was %d" % return_value)
        return status

    def _run_glp_mip(self):
        return_value = glp_intopt(self.problem, self.configuration._iocp)
        glpk_status = glp_mip_status(self.problem)
        if return_value == 0:
            status = _GLPK_STATUS_TO_STATUS[glpk_status]
        elif return_value == GLP_ETMLIM:
            status = interface.TIME_LIMIT
        else:
            status = _GLPK_STATUS_TO_STATUS[glpk_status]
            if status == interface.UNDEFINED:
                log.debug("Status undefined. GLPK status code returned by glp_intopt was %d" % return_value)
        return status

    def _optimize(self):
        status = self._run_glp_simplex()

        if status == interface.UNDEFINED and self.configuration.presolve is True:
            # If presolve is on, status will be undefined if not optimal
            self.configuration.presolve = False
            status = self._run_glp_simplex()
            self.configuration.presolve = True
        if (glp_get_num_int(self.problem) + glp_get_num_bin(self.problem)) > 0:
            status = self._run_glp_mip()
            if status == 'undefined' or status == 'infeasible':
                # Let's see if the presolver and some scaling can fix this issue
                glp_scale_prob(self.problem, GLP_SF_AUTO)
                original_presolve_setting = self.configuration.presolve
                self.configuration.presolve = True
                status = self._run_glp_mip()
                self.configuration.presolve = original_presolve_setting
        self._status = status
        return status

    def _add_variables(self, variables):
        for variable in variables:
            glp_add_cols(self.problem, 1)
            index = glp_get_num_cols(self.problem)
            glp_set_col_name(self.problem, index, str(variable.name))
            variable.problem = self
            self._glpk_set_col_bounds(variable)
            glp_set_col_kind(self.problem, variable.index, _VTYPE_TO_GLPK_VTYPE[variable.type])
        super(Model, self)._add_variables(variables)

    def _remove_variables(self, variables):
        if len(variables) > 0:
            if len(variables) > 350:
                delete_indices = [variable.index - 1 for variable in variables]
                keep_indices = [i for i in range(0, len(self.variables)) if i not in delete_indices]
                self._variables = self.variables.fromkeys(keep_indices)
            else:
                for variable in variables:
                    del self._variables[variable.name]

            num = intArray(len(variables) + 1)
            for i, variable in enumerate(variables):
                num[i + 1] = variable.index
            glp_del_cols(self.problem, len(variables), num)

            for variable in variables:
                del self._variables_to_constraints_mapping[variable.name]
                variable.problem = None

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)
        for constraint in constraints:
            constraint._problem = None  # This needs to be dones in order to not trigger constraint._get_expression()
            glp_add_rows(self.problem, 1)
            index = glp_get_num_rows(self.problem)
            glp_set_row_name(self.problem, index, str(constraint.name))
            num_cols = glp_get_num_cols(self.problem)
            index_array = intArray(num_cols + 1)
            value_array = doubleArray(num_cols + 1)
            num_vars = 0  # constraint.variables is too expensive for large problems
            if constraint.expression.is_Atom and constraint.expression.is_Symbol:
                var = constraint.expression
                index_array[1] = var.index
                value_array[1] = 1
                num_vars += 1
            elif constraint.expression.is_Mul:
                args = constraint.expression.args
                if len(args) > 2:
                    raise Exception(
                        "Term(s) %s from constraint %s is not a proper linear term." % (args, constraint))
                coeff = float(args[0])
                var = args[1]
                index_array[1] = var.index
                value_array[1] = coeff
                num_vars += 1
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
                    num_vars += 1
            glp_set_mat_row(self.problem, index, num_vars,
                            index_array, value_array)
            constraint._problem = self
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
        elif constraint.lb == constraint.ub:
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

    def _remove_constraints(self, constraints):
        if len(constraints) > 0:
            constraint_indices = [constraint.index for constraint in constraints]
            super(Model, self)._remove_constraints(constraints)
            num = intArray(len(constraints) + 1)
            for i, constraint_index in enumerate(constraint_indices):
                num[i + 1] = constraint_index
            glp_del_rows(self.problem, len(constraints), num)

    def _set_linear_objective_term(self, variable, coefficient):
        glp_set_obj_coef(self.problem, variable.index, coefficient)


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
    print("status:", model.status)
    print("objective value:", model.objective.value)

    for var_name, var in model.variables.items():
        print(var_name, "=", var.primal)

    print(model)

    from swiglpk import glp_read_lp

    problem = glp_create_prob()
    glp_read_lp(problem, None, "../tests/data/model.lp")

    solver = Model(problem=problem)
    print(solver.optimize())
    print(solver.objective)

    import time

    t1 = time.time()
    print("pickling")
    pickle_string = pickle.dumps(solver)
    resurrected_solver = pickle.loads(pickle_string)
    t2 = time.time()
    print("Execution time: %s" % (t2 - t1))

    resurrected_solver.optimize()
    print("Halelujah!", resurrected_solver.objective.value)
