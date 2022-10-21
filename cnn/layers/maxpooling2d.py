import cnn.base.layer as Layer
import numpy as np

class MaxPoolingLayer2D(Layer):
    def __init__(self, pool_size, stride_pooling, keepDims = False, stride_padding = [20, 16], kernel_size = [10,8], name = None):
        if isinstance(pool_size, (np.ndarray, list, tuple)):
            if not all(pool_size) or sum(1 for param in pool_size if param < 0) > 1:
                raise ValueError(      
                    "Expected only positive, non zero integers as pool size\n"
                    f"Got: pool_size={pool_size}."
                )
            else:
                pool_size = np.array(pool_size)
        else:
            if isinstance(pool_size, int):
                if not pool_size > 0:
                    raise ValueError(      
                        "Expected only positive, non zero integers as pool size\n"
                        f"Got: pool_size={pool_size}."
                    )
                else:
                    pool_size = np.array([pool_size, pool_size])
            else:
                raise ValueError(      
                    "Expected only positive, non zero integers as pool size\n"
                    f"Got: pool_size={pool_size}."
                )

        if isinstance(stride_padding, (np.ndarray, list, tuple)):
            if not all(stride_padding) or sum(1 for param in stride_padding if param < 0) > 1:
                raise ValueError(      
                    "Expected only positive, non zero integers as pool size\n"
                    f"Got: stride_padding={stride_padding}."
                )
            else:
                stride_padding = np.array(stride_padding)
        else:
            if isinstance(stride_padding, int):
                if not stride_padding > 0:
                    raise ValueError(      
                        "Expected only positive, non zero integers as pool size\n"
                        f"Got: stride_padding={stride_padding}."
                    )
                else:
                    stride_padding = np.array([stride_padding, stride_padding])
            else:
                raise ValueError(      
                    "Expected only positive, non zero integers as pool size\n"
                    f"Got: pool_size={stride_padding}."
                )

        if isinstance(kernel_size, (np.ndarray, list, tuple)):
            if not all(kernel_size) or sum(1 for param in kernel_size if param < 0) > 1:
                raise ValueError(      
                    "Expected only positive, non zero integers as pool size\n"
                    f"Got: stride_padding={kernel_size}."
                )
            else:
                kernel_size = np.array(kernel_size)
        else:
            if isinstance(kernel_size, int):
                if not kernel_size > 0:
                    raise ValueError(      
                        "Expected only positive, non zero integers as pool size\n"
                        f"Got: stride_padding={kernel_size}."
                    )
                else:
                    kernel_size = np.array([kernel_size, kernel_size])
            else:
                raise ValueError(      
                    "Expected only positive, non zero integers as pool size\n"
                    f"Got: pool_size={kernel_size}."
                )

        if isinstance(stride_pooling, (np.ndarray, list, tuple)):
            if not all(stride_pooling) or sum(1 for param in stride_pooling if param < 0) > 1:
                raise ValueError(      
                    "Expected only positive, non zero integers as stride_pooling\n"
                    f"Got: stride_pooling={stride_pooling}."
                )
            else:
                stride_pooling = np.array(stride_pooling)
        else:
            if isinstance(stride_pooling, int):
                if not stride_pooling > 0:
                    raise ValueError(      
                        "Expected only positive, non zero integers as stride_pooling\n"
                        f"Got: stride_pooling={stride_pooling}."
                    )
                else:
                    stride_pooling = np.array([stride_pooling, stride_pooling])
            else:
                raise ValueError(      
                    "Expected only positive, non zero integers as stride\n"
                    f"Got: stride_pooling={stride_pooling}."
                    )

        if not isinstance(keepDims, str):
            if keepDims not in (True, False):
                raise NotImplementedError(
                    "Expected true or false\n"
                    f"Got: {keepDims}"
                )
        else:
            raise NotImplementedError(
                    "Expected true or false\n"
                    f"Got: {keepDims}"
                )

        if not isinstance(name, (str, type(None))):
            raise ValueError(      
            "Layer-name needs to be a string!\n"
            f"Got: Layer-name={name}."
            )

        self.keepDims = keepDims
        self.stride_pooling = stride_pooling
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.stride_padding = stride_padding
        self.name = name

    def maxPooling2d(self, data):
        pooled_output = []
        img = data

        if self.keepDims:
            img = Layer.addPadding(data, self.kernel_size, self.stride_padding)
            
        output_shape = Layer.calc_output_size_padMode(img.shape, self.pool_size, self.stride_pooling, 'valid')

        if len(img.shape) == 2:
            for i in np.arange(img.shape[0], step=self.stride_pooling[0]):
                for j in np.arange(img.shape[1], step=self.stride_pooling[1]):
                    temp = img[i:i+self.pool_size[0], j:j+self.pool_size[1]]  
                    if temp.shape == tuple(self.pool_size):
                        pooled_output.append(np.max(temp))
            
            pooled_output = np.array(pooled_output).reshape(output_shape)
        
        else:
            for z in range(len(img.shape)):
                temp2 = img[:,:,z]
                temp3 = []

                for i in np.arange(temp2.shape[0], step=self.stride_pooling[0]):
                    for j in np.arange(temp2.shape[1], step=self.stride_pooling[1]):
                        temp = temp2[i:i+self.pool_size[0], j:j+self.pool_size[1]]  
                        if temp.shape == tuple(self.pool_size):
                            temp3.append(np.max(temp))
                
                pooled_output.append(np.array(temp3).reshape(output_shape))
            pooled_output = np.dstack(pooled_output)

        return pooled_output

    def forward(self, data):
        return self.maxPooling2d(data)
