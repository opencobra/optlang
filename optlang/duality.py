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

from sympy import Add, Mul


# This function is very complex. Should maybe be refactored
def convert_linear_problem_to_dual(model, sloppy=False, infinity=None, maintain_standard_form=True, prefix="dual_", dual_model=None):  # NOQA
    """
    A mathematical optimization problem can be viewed as a primal and a dual problem. If the primal problem is
    a minimization problem the dual is a maximization problem, and the optimal value of the dual is a lower bound of
    the optimal value of the primal.
    For linear problems, strong duality holds, which means that the optimal values of the primal and dual are equal
    (duality gap = 0).

    This functions takes an optlang Model representing a primal linear problem and returns a new Model representing
    the dual optimization problem. The provided model must have a linear objective, linear constraints and only
    continuous variables. Furthermore, the problem must be in standard form, i.e. all variables should be non-negative.
    Both minimization and maximization problems are allowed. The objective direction of the dual will always be
    opposite of the primal.

    Attributes:
    ----------
    model: optlang.interface.Model
        The primal problem to be dualized
    sloppy: Boolean (default False)
        If True, linearity, variable types and standard form will not be checked. Only use if you know the primal is
        valid
    infinity: Numeric or None
        If not None this value will be used as bounds instead of unbounded variables.
    maintain_standard_form: Boolean (default True)
        If False the returned dual problem will not be in standard form, but will have fewer variables and/or constraints
    prefix: str
        The string that will be prepended to all variable and constraint names in the returned dual problem.
    dual_model: optlang.interface.Model or None (default)
        If not None, the dual variables and constraints will be added to this model. Note the objective will also be
        set to the dual objective. If None a new model will be created.

    Returns:
    ----------
    dual_problem: optlang.interface.Model (same solver as the primal)
    """
    if dual_model is None:
        dual_model = model.interface.Model()

    maximization = model.objective.direction == "max"

    if infinity is not None:
        neg_infinity = -infinity
    else:
        neg_infinity = None

    if maximization:
        sign = 1
    else:
        sign = -1

    coefficients = {}
    dual_objective = {}

    # Add dual variables from primal constraints:
    for constraint in model.constraints:
        if constraint.expression == 0:
            continue  # Skip empty constraint
        if not (sloppy or constraint.is_Linear):
            raise ValueError("Non-linear problems are not supported: " + str(constraint))
        if constraint.lb is None and constraint.ub is None:
            continue  # Skip free constraint
        if not maintain_standard_form and constraint.lb == constraint.ub:
            const_var = model.interface.Variable(prefix + constraint.name + "_constraint", lb=neg_infinity, ub=infinity)
            dual_model.add(const_var)
            if constraint.lb != 0:
                dual_objective[const_var] = sign * constraint.lb
            for variable, coef in constraint.expression.as_coefficients_dict().items():
                coefficients.setdefault(variable.name, {})[const_var] = sign * coef
        else:
            if constraint.lb is not None:
                lb_var = model.interface.Variable(prefix + constraint.name + "_constraint_lb", lb=0, ub=infinity)
                dual_model.add(lb_var)
                if constraint.lb != 0:
                    dual_objective[lb_var] = -sign * constraint.lb
            if constraint.ub is not None:
                ub_var = model.interface.Variable(prefix + constraint.name + "_constraint_ub", lb=0, ub=infinity)
                dual_model.add(ub_var)
                if constraint.ub != 0:
                    dual_objective[ub_var] = sign * constraint.ub

            assert constraint.expression.is_Add or constraint.expression.is_Mul, \
                "Invalid expression type: " + str(type(constraint.expression))
            if constraint.expression.is_Add:
                coefficients_dict = constraint.expression.as_coefficients_dict()
            else:  # constraint.expression.is_Mul:
                coefficients_dict = {constraint.expression.args[1]: constraint.expression.args[0]}

            for variable, coef in coefficients_dict.items():
                if constraint.lb is not None:
                    coefficients.setdefault(variable.name, {})[lb_var] = -sign * coef
                if constraint.ub is not None:
                    coefficients.setdefault(variable.name, {})[ub_var] = sign * coef

    # Add dual variables from primal bounds
    for variable in model.variables:
        if not (sloppy or variable.type == "continuous"):
            raise ValueError("Integer variables are not supported: " + str(variable))
        if not sloppy and (variable.lb is None or variable.lb < 0):
            raise ValueError("Problem is not in standard form (" + variable.name + " can be negative)")
        if variable.lb > 0:
            bound_var = model.interface.Variable(prefix + variable.name + "_lb", lb=0, ub=infinity)
            dual_model.add(bound_var)
            coefficients.setdefault(variable.name, {})[bound_var] = -sign * 1
            dual_objective[bound_var] = -sign * variable.lb
        if variable.ub is not None:
            bound_var = model.interface.Variable(prefix + variable.name + "_ub", lb=0, ub=infinity)
            dual_model.add(bound_var)
            coefficients.setdefault(variable.name, {})[bound_var] = sign * 1
            if variable.ub != 0:
                dual_objective[bound_var] = sign * variable.ub

    # Add dual constraints from primal objective
    primal_objective_dict = model.objective.expression.as_coefficients_dict()
    for variable in model.variables:
        expr = Add(*((coef * dual_var) for dual_var, coef in coefficients[variable.name].items()))
        obj_coef = primal_objective_dict[variable]
        if maximization:
            const = model.interface.Constraint(expr, lb=obj_coef, name=prefix + variable.name)
        else:
            const = model.interface.Constraint(expr, ub=obj_coef, name=prefix + variable.name)
        dual_model.add(const)

    # Make dual objective
    expr = Add(*((coef * dual_var) for dual_var, coef in dual_objective.items() if coef != 0))
    if maximization:
        objective = model.interface.Objective(expr, direction="min")
    else:
        objective = model.interface.Objective(expr, direction="max")
    dual_model.objective = objective

    return dual_model
