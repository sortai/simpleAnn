import numpy as np
from difun import difun, Dfuns #differentiable functions + premade

class SizeMismatch (Exception):
    pass

class layer:
    pass

class llayer (layer):
    def __init__(self, size, w=None, b=None):
        self.size = tuple(size)
        if w is None:
            w = np.zeros((self.size[1],self.size[0]))
        if b is None:
            b = np.ones(self.size[1])
        self.w = np.array(w)
        self.b = np.array(b)
    def __call__(self, i):
        return np.matmul(self.w, i)+self.b
    def dei(self, i):
        return self.w.copy()

class nlayer (layer):
    def __init__(self, size, fun=None):
        try: self.size = tuple(size)
        except TypeError: self.size = (size, size)
        if fun is None:
            fun = Dfuns["lReLU"] #default
        self.fun = fun
    def __call__(self, i):
        return self.fun(np.array(i))
    def dei(self, i):
        i=np.array(i)
        if (not isinstance(i.size, int)) or i.size!=self.size[0]: raise SizeMismatch("nlayer.dei(self, {})".format(i))
        return self.fun.de(i)

class ann (llayer):
    def __init__(self, layers=None, size=None):
        if size is None: size=(layers[0].size[0], layers[-1].size[1])
        self.size = tuple(size)
        if layers is None:
            self.__monolayer=True
            super().__init__(self.size)
        else:
            self.__monolayer=False
            if len(layers) < 1:
                raise SizeMismatch("not enough layers provided")
            if self.size[0]!=layers[0].size[0]:
                raise SizeMismatch("The input size of the network doesn't match with the input size of the first layer")
            if self.size[1]!=layers[-1].size[1]:
                raise SizeMismatch("The output size of the network doesn't match with the output size of the last layer")
            self.layers = layers
            self.sizes=[self.size[0]]
            for ln, la in enumerate(self.layers[:-1]):
                if la.size[1]!=self.layers[ln+1].size[0]:
                    raise SizeMismatch("Layer output size of layer {ln0} ({so}) and layer input size of layer {ln1} ({si}) don't match".format(ln0=ln, ln1=ln+1, so=la.size[1], si=self.layers[ln+1].size[0]))
                else: self.sizes.append(la.size[1])
            self.sizes.append(self.size[1])
            self.sizes = tuple(self.sizes)
    @property
    def monolayer(self):
        return self.__monolayer
    @monolayer.setter
    def monolayer(self, m):
        raise AttributeError('You can\'t actually change the "monolayer" property directly')
    def __call__(self, i):
        if self.__monolayer:
            return super().__call__(i)
        for layer in self.layers:
            i=layer(i)
        return i
##    def dei(self, i):
##        de=self.layers[0].dei(i)
##        i=self.layers[0](i)
##        for layer in self.layers[1:]:
##            de=np.matmul(de.swapaxes(0,1), layer(i))
##            i=layer(i)
##        return de
            
