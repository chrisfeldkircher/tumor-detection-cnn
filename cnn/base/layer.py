import numpy as np

class Layer:
    def __init__(self):
        self.kernels = None
        
    @staticmethod
    def calcPadding(input_size, kernel_size, stride): 
        remainder = None
        if all(isinstance(i, (list, tuple, np.ndarray)) for i in [input_size, kernel_size, stride]):
            input_size = np.array(input_size)
            kernel_size = np.array(kernel_size)
            stride = np.array(stride)
            #Convert the inputs to numpy arrays to avoid type errors

            remainder = ((input_size-kernel_size) % stride)
            #Check if we need padding
            nopadding = all(v == 0 for v in remainder)
        
        elif all(isinstance(i, int) for i in [input_size, kernel_size, stride]):
            remainder = ((input_size-kernel_size) % stride)

            nopadding = remainder == 0
        
        else:
            raise ValueError(      
                "Input_size, kernel_size and stride have to be the same type!"
            )
        if nopadding:
            return (input_size - input_size).astype(int) #to account for integers, lists and touples
            #[[x], [y]] -> no pdding needed
        else:
            return ((remainder/np.max(remainder))*kernel_size - (remainder/np.max(remainder))).astype(int)
            #[[x], [y]] -> padding needed: filtersize - 1

    @staticmethod
    def calc_output_size_padMode(input_size, kernel_size, stride, padding):
        full = []
        valid = []

        #gray scaled data
        if len(input_size) > 2:
            input_size = (input_size[0], input_size[1])
        
        pad = Layer.calcPadding(input_size, kernel_size, stride)

        valid = ((input_size - kernel_size + 1) + stride - 1) // stride
        full = ((input_size - kernel_size+pad)//stride) + 1
        #full = ((input_size + kernel_size - 1) + stride - 1) // stride

        if padding.lower() == 'valid':
            return valid
        elif padding.lower() == 'full':
            return full

    @staticmethod
    def addPadding(data, kernel_size, stride):
        paddedData = []
        temp = []

        #check if only gray scaled data
        if len(data.shape) == 2:
            paddingcoords = Layer.calcPadding(data.shape, kernel_size, stride)
            paddedData = np.pad(data, ((0, int(paddingcoords[0])), (0, int(paddingcoords[1]))))
        else:
            for i in range(len(data.shape)):
                temp = data[:,:,i]
                paddingcoords = Layer.calcPadding(temp.shape, kernel_size, stride)
                paddedData.append(np.pad(temp, ((0, int(paddingcoords[0])), (0, int(paddingcoords[1])))))
            
            paddedData = np.dstack(paddedData)

        return paddedData

    @staticmethod
    def outputLayerVolume(input_layer, prev_layer, stride, padding):
        vol_input_layer = input_layer.depth * input_layer.height * input_layer.width
        vol_prev_layer = prev_layer.depth * prev_layer.height * prev_layer.width

        vol_output_layer = (vol_input_layer - vol_prev_layer + 2*padding)/stride + 1

        return vol_output_layer