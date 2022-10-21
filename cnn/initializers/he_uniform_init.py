import cnn.base.initialization as Initialization
import numpy as np

# Initializer for reLU type activation functions
class HeUniformInitialization(Initialization):
    def initialize(depth, kernel_w, kernel_h, channels, seed = None):
        fan_in = kernel_h * kernel_w

        if seed is not None:
            np.random.seed(seed)

        weights = np.random.uniform(0.0, low=-np.sqrt(6 / float(fan_in)), high=np.sqrt(6 / float(fan_in)), size=(depth, kernel_w, kernel_h, channels))
        
        return weights