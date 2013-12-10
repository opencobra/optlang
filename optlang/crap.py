import sympy

class Test(object):
    """docstring for Test"""
    def __init__(self):
        super(Test, self).__init__()
        self.blub = 'blub'
        self.expr = [1,2,3,4]

    def __getattr__(self, name):
        print name
        return getattr(self.expr, name)

    def __dir__(self):
        return self.__dict__.keys() + dir(self.expr)

class Test2(Test):
    def __init__(self):
        super(Test2, self).__init__()

    # def __getattr__(self, name):
    #     print name + "Hongo"
    #     return getattr(self.expr, name)


class Variable(sympy.Symbol):

    """docstring for Column"""

    def __init__(self, name, lb=None, ub=None, type="continuous", problem=None, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)
        self._lb = lb
        self._ub = ub
        self._type = type
        self._problem = problem
        self.john = 'john'

    @property
    def lb(self):
        return self._lb
    @lb.setter
    def lb(self, value):
        if self.ub is not None and self.ub < value:
            raise Exception("Lower bound ...")
        self._lb = value

    @property
    def ub(self):
        return self._ub
    @ub.setter
    def ub(self, value):
        if self.lb is not None and self.lb > value:
            raise Exception("Upper bound ...")
        self._ub = value
    
    @property
    def type(self):
        return self._type
    @type.setter
    def type(self, value):
        self._type = value

    @property
    def problem(self):
        return self._problem
    @problem.setter
    def problem(self, value):
        self._problem = value


class Variable2(Variable):
    """..."""

    def __init__(self, *args, **kwargs):
        super(Variable2, self).__init__(*args, **kwargs)

    @property
    def lb(self):
        print 'asdfasdf'
        # return super(Variable, self).lb.fget(self)
        return Variable.lb.fget(self)
    @lb.setter
    def lb(self, value):
        print 'Hey ho ho'
        # super(Variable, self).lb.fset = value
        Variable.lb.fset(self, value)
    
# solution = interface.Solution()
        # glp_simplex(self.problem, self._smcp)
        # glpk_status = glp_get_status(self.problem)
        # self.status = self._glpk_status_to_status[glpk_status]
        # solution.status = self.status
        # solution.objval = glp_get_obj_val(self.problem)
        # for var_id, variable in self.variables.items():
        #     variable.primal = glp_get_col_prim(self.problem, variable.index)
        #     variable.dual = glp_get_col_dual(self.problem, variable.index)
        #     solution.variables[var_id]  = {'primal': variable.primal, 'dual': variable.dual}
        # for constr_id, constraint in self.constraints.items():
        #     constraint.primal = glp_get_row_prim(self.problem, constraint.index)
        #     constraint.dual = glp_get_row_dual(self.problem, constraint.index)
        #     solution.constraints[constr_id]  = {'primal': constraint.primal, 'dual': constraint.dual}
        # return solution


if __name__ == '__main__':
    var1 = Variable('test1', lb=10, ub=20)
    var2 = Variable2('test2', lb=10, ub=20)
    print var1, var2

    test = Test()
    print test.blub
    print test.append