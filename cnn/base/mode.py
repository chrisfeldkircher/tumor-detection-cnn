import layer as Layer
import activation as Activation
import numpy as np

class Model:
    def __init__(self):
        self.layers = {} #dict of layers
        self.trained_model = None #trained_Model object
        self.history = []

    def add(self, layer):
        if not isinstance(layer, (Layer, Activation)):
            raise TypeError(
                "Expected layer of layer class\n"
                f"Got: layer={layer} of type {type(layer)}"
            )
        else:
            order = len(self.layers.keys()) #get the input order

            if layer.name == None:
                count = len([x for x in list(self.layers.keys()) if x.split('-')[0] == str(type(layer).__name__)]) #Get amount of layer-types

                self.layers[f'{type(layer).__name__}-{count}-{order}'] = layer
            else: 
                self.layers[f'{type(layer).__name__}-{layer.name}-{order}'] = layer

    def forward(self, init_data):
        data = []

        self.history = sorted(list(self.layers.keys()), key=lambda x: x.rsplit('-')[-1]) #get execution order

        X = self.layers[self.history[0]].forward(data = init_data, first = True)

        for i in range(1, len(self.history)):
            X = self.layers[self.history[i]].forward(data = X)
        
        #return np.array(X)/255
        for elem in X:
            data.append(elem/np.max(elem))
        return np.array(data)

    def train_model(self, epochs, loss, optimizer, metrics):
        pass

    def predict(self):
        pass