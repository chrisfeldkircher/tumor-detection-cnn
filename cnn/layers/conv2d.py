import cnn.base.layer as Layer
import numpy as np

#TODO: add dilitation option for filters
class ConvLayer2D(Layer):
    def __init__(self, num_filter, kernel, padding, stride, name = None, initilizer = None):
        if isinstance(kernel, (np.ndarray, list)):
            self.kernel = kernel
            self.kernel_size = np.array(kernel.shape)
            self.provided_kernel = True
        elif isinstance(kernel, int):
            self.kernel_size = np.array([kernel, kernel])
            self.provided_kernel = False
        else:
            raise ValueError(      
                "Expected a list or numpy-array as kernel or int as kernelsize\n"
                f"Got: kernel={kernel}."
            )

        '''
        if isinstance(initilizer, Initialization):
            self.initilizer = initilizer
            if 'seed' in initilizer.__code__.co_varnames:
                #self.seed = 
                pass
        else:
            raise ValueError(      
                "Expected a list or numpy-array as kernel or int as kernelsize\n"
                f"Got: kernel={kernel}."
            )
        '''

        if isinstance(num_filter, int):
            if not num_filter > 0:
                raise ValueError(      
                    "Expected only positive, non zero integer as num. filters\n"
                    f"Got: num_filter={num_filter}."
                )
            else:
                self.depth = num_filter
        else:
            raise ValueError(      
                "Expected only positive, non zero integer as num. filters\n"
                f"Got: num_filter={num_filter}."
            )

        if isinstance(padding, str):
            if padding.lower() not in ("full", "valid"):
                raise NotImplementedError(
                    f"The mode: {padding.lower()} is not supported\n"
                    "Please select one of the following modes: valid, full"
                )
            else:
                self.padding = padding.lower()
        else:
            raise NotImplementedError(
                "Tuples, lists and integers a currently not supported for padding\n"
                "Please select one of the follwing modes: valid, full"
                )

        if isinstance(stride, (np.ndarray, list, tuple)):
            if not all(stride) or sum(1 for param in stride if param < 0) > 1:
                raise ValueError(      
                    "Expected only positive, non zero integers as stride\n"
                    f"Got: stride={stride}."
                )
            else:
                self.stride = np.array(stride)
        else:
            if isinstance(stride, int):
                if not stride > 0:
                    raise ValueError(      
                        "Expected only positive, non zero integers as stride\n"
                        f"Got: stride={stride}."
                    )
                else:
                    self.stride = np.array([stride, stride])
            else:
                raise ValueError(      
                    "Expected only positive, non zero integers as stride\n"
                    f"Got: stride={stride}."
                    )

        if not isinstance(name, (str, type(None))):
            raise ValueError(      
            "Layer-name needs to be a string!\n"
            f"Got: Layer-name={name}."
            )

        self.name = name
    
    #@jit(forceobj=True)
    def convolve2D(self, data, kernel = None, first_layer = False):
        img = data
        convolved_output = []

        if first_layer:
            if kernel is None:
                if len(data.shape) > 2: #colored image
                    #kernel = self.initilizer.initialize(self.depth, self.kernel_size[0], self.kernel_size[1],data.shape[2], self.seed)
                    Layer.kernels = np.random.randn(self.depth, self.kernel_size[0], self.kernel_size[1], data.shape[2]) 
                    #shape: filter_depth x kernel_size x kernel_size x color_channels
                else: #gray scaled image
                    #kernel = self.initilizer.initialize(self.depth, self.kernel_size[0], self.kernel_size[1] ,data.shape[2], self.seed)
                    Layer.kernels = np.random.randn(self.depth, self.kernel_size[0], self.kernel_size[1], 1)
                    #shape: filter_depth x kernel_size x kernel_size x color_channels

            else:
                Layer.kernels = self.kernel
            
            output_shape = Layer.calc_output_size_padMode(data.shape, self.kernel_size, self.stride, self.padding)

        else:
            output_shape = Layer.calc_output_size_padMode(data[0].shape, self.kernel_size, self.stride, self.padding)

        if self.padding == 'full':
            if isinstance(data, np.ndarray):
                img = Layer.addPadding(data, self.kernel_size, self.stride)
            else:
                img = list(img)
                for elem in data:
                    img.append(Layer.addPadding(elem, self.kernel_size, self.stride))

        if first_layer:
            if len(img.shape) == 2:
                for kernel in Layer.kernels:
                    temp2 = []
                    for i in np.arange(img.shape[0], step=self.stride[0]):
                        for j in np.arange(img.shape[1], step=self.stride[1]):
                            temp = img[i:i+self.kernel_size[0], j:j+self.kernel_size[1]]  
                            if temp.shape == tuple(self.kernel_size):
                                #temp2.append(signal.convolve2d(temp, kernel, mode='valid'))
                                temp2.append((temp*kernel).sum())
                                #temp2.append(signal.correlate2d(temp, kernel,'valid'))
                    conv_temp = np.array(temp2).reshape(output_shape)
                    conv_temp = conv_temp/np.max(conv_temp)
                    convolved_output.append(conv_temp)
            else:
                for kernel in Layer.kernels:
                    temp4 = []
                    for z in range(len(img.shape)):
                        temp2 = img[:,:,z]
                        temp3 = []

                        for i in np.arange(temp2.shape[0], step=self.stride[0]):
                            for j in np.arange(temp2.shape[1], step=self.stride[1]):
                                temp = temp2[i:i+self.kernel_size[0], j:j+self.kernel_size[1]]  
                                if temp.shape == tuple(self.kernel_size):
                                    #temp3.append(signal.convolve2d(temp, kernel, mode='valid'))
                                    temp3.append((temp*kernel).sum())
                                    #temp3.append(signal.correlate2d(temp, kernel,'valid'))

                        temp4.append(np.array(temp3).reshape(output_shape))

                    convolved_output.append(np.dstack(temp4))
        
        else:
            if len(img[0].shape) == 2:
                for k in range(len(Layer.kernels)):
                    temp2 = []
                    for i in np.arange(img[k].shape[0], step=self.stride[0]):
                        for j in np.arange(img[k].shape[1], step=self.stride[1]):
                            temp = img[k][i:i+self.kernel_size[0], j:j+self.kernel_size[1]]  
                            if temp.shape == tuple(self.kernel_size):
                                #temp2.append(signal.convolve2d(temp, kernel, mode='valid'))
                                temp2.append((temp*Layer.kernels[k]).sum())
                                #temp2.append(signal.correlate2d(temp, kernel,'valid'))
                    conv_temp = np.array(temp2).reshape(output_shape)
                    conv_temp = conv_temp/np.max(conv_temp)
                    convolved_output.append(conv_temp)

            else:
                for k in range(len(Layer.kernels)):
                    temp4 = []
                    for z in range(len(img[k].shape)):
                        temp2 = img[k][:,:,z]
                        temp3 = []

                        for i in np.arange(temp2.shape[0], step=self.stride[0]):
                            for j in np.arange(temp2.shape[1], step=self.stride[1]):
                                temp = temp2[i:i+self.kernel_size[0], j:j+self.kernel_size[1]]  
                                if temp.shape == tuple(self.kernel_size):
                                    #temp3.append(signal.convolve2d(temp, kernel, mode='valid'))
                                    temp3.append((temp*Layer.kernels[k]).sum())
                                    #temp3.append(signal.correlate2d(temp, kernel,'valid'))

                        temp4.append(np.array(temp3).reshape(output_shape))
                    convolved_output.append(np.dstack(temp4))
        
        return convolved_output

    def forward(self, data, first = False):
        if first and self.provided_kernel:
            return self.convolve2D(data = data, kernel = self.kernel, first_layer = True)

        elif first and not self.provided_kernel:
            return self.convolve2D(data = data, first_layer = True)
        
        elif not first and self.provided_kernel:
            return self.convolve2D(data = data, kernel = self.kernel)

        return self.convolve2D(data = data)