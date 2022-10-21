import cnn.base.layer as Layer
import numpy as np

class DropoutLayer2D(Layer):
    def __init__(self, probability):
        if isinstance(probability, float):
            if probability > 0:
                self.probability = probability
            else:
                raise ValueError(      
                        "Expected a positive, non zero float number as probability\n"
                        f"Got: probability={probability}."
                    )
        else:
            raise ValueError(      
                    "Expected a positive, non zero float number as probability\n"
                    f"Got: probability={probability}."
                )

    def dropout2D(self, data):
        X = data
        X *= (1. - self.probability)
        return X

    def foward(self, data):
        return self.dropout2D(data)
        