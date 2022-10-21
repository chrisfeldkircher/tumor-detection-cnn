import cnn.base.initialization as Initialization
import numpy as np

# Initializer for sigmoid/ logistic type activation functions
class GlorotNormalInitialization(Initialization):
    def initialize(depth, kernel_w, kernel_h, channels, seed = None):
        fan_in = kernel_h * kernel_w
        fan_out = kernel_h * kernel_w * channels

        if seed is not None:
            np.random.seed(seed)

        weights = np.random.normal(0.0, np.sqrt(2 / float(fan_in + fan_out)), size=(depth, kernel_w, kernel_h, channels))
        
        return weights