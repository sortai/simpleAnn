import numpy as np

class difun:
    def __init__(self, fun, funp=None, p=2**-10):
        self.__fun = fun
        self.__funp = funp
        if self.__funp is not None: p = 0
        self.__p=p
    @property
    def p(self):
        return self.__p
    @p.setter
    def p(self, p):
        if self.__funp is None: self.__p=p
    def __call__(self, i):
        return self.__fun(i)
    def de(self, i):
        if self.__funp is not None: return self.__funp(i)
        return (self(i+self.__p)-self(i))/self.__p

Dfuns = {
    "i": difun(lambda x: x, lambda x: x*0+1),
    "ReLU": difun(lambda x: np.maximum(0, x)),
    "lReLU": difun(lambda x: np.maximum(x, x*.1)),
    "sin": difun(lambda x: np.sin(x), lambda x: np.cos(x)),
    "cos": difun(lambda x: np.cos(x), lambda x: -np.sin(x)),
    "gauss": digfun(lambda x: np.exp(-(x**2)), lambda x: -2*x*np.exp(-(x**2))),
    "softplus": difun(lambda x: np.ln(np.exp(x)+1)),
    }

