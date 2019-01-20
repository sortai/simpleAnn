import numpy as np


class SizeMismatch (Exception):
    pass

class layer:
    def __init__(self, size):
        pass

class ann (layer):
    def __init__(self, size, layers=None):
        if layers is None:
            self._monolayer=True
            super().__init__(self, size)
        else:
            self._monolayer=False
            if len(layers) < 1:
                raise SizeMismatch("not enough layers provided")
            if size[0]!=layers[0].size[0]:
                raise SizeMismatch("The input size of the network doesn't match with the input size of the first layer")
            if size[1]!=layers[-1].size[2]:
                raise SizeMismatch("The output size of the network doesn't match with the output size of the last layer")
            self.size = size
            self.layers = layers
            self.sizes=[self.size[0]]
            for ln, la in enumerate(self.layers[:-1]):
                if la.size[1]!=self.layers[ln+1].size[0]:
                    raise SizeMismatch("Layer output size of layer {ln0} ({so}) and layer input size of layer {ln1} ({si}) don't match".format(ln0=ln, ln1=ln+1, so=la.size[1], si=self.layers[ln+1].size[0]))
                else: self.sizes.append(la.size[1])
            self.sizes.append(self.size[1])
    @property
    def monolayer(self):
        return self._monolayer
    def __call__(self, i):
        if self._monolayer:
            return super().__call__(self, i)
        for layer in self.layers:
            i=layer(i)
        return i
