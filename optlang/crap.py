import sympy

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
    


if __name__ == '__main__':
    var1 = Variable('test1', lb=10, ub=20)
    var2 = Variable2('test2', lb=10, ub=20)
    print var1, var2

