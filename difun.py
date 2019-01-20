import numpy as np

class difun:
    def __init__(self, fun, funp=None, p=2**-10):
        self._fun = fun
        self._funp = funp
        self._p = p
