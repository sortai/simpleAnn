
class layer:
    def __init__(self, size):
        pass

class ann (layer):
    def __init__(self, size, layers=None):
        if layers is None:
            super().__init__(self, size)
