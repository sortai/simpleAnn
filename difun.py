import numpy as np

class difun:
    def __init__(self, fun, funp=None, p=2**-10):
        self.__fun = fun
        self.__funp = funp
        self.p = p
    def __call__(self, i):
        return self.__fun(i)
    def de(self, i):
        try: return self.__funp(i)
        except TypeError as error:
            if error.args[0] == "'NoneType' object is not callable":
                return (self(i+self.p)-self(i))/self.p

Dfuns = {
    "lReLU": difun(lambda x: max(x,x*.1)),
    }
