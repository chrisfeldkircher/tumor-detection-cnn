import cnn.base.activation as Activation
import numpy as np

class ReLU_activation(Activation):
    def __init__(self, name=None):
        self.name = name

    def forward(self, data):
        return np.maximum(0, data) #maximum of either 0 or i; if i > 0: append(i) else: append(0)