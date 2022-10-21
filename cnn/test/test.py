import convolutional_neural_network as cnn
import numpy as np
import cv2
import time

img_letter = cv2.imread("C:\\Users\\Christoph Feldkirchn\\OneDrive - McKinsey & Company\\Desktop\\Programming\\Neural Networks\\cnn\\679-6796443_letter-t-png-normal-letter-t-transparent.png")
img_pan = cv2.imread("C:\\Users\\Christoph Feldkirchn\\OneDrive - McKinsey & Company\\Desktop\\Programming\\Neural Networks\\cnn\\istockphoto-1145618475-612x612.jpg")

# loads it in BRG instead of RGB
#img_letter = cv2.cvtColor(img_letter, cv2.COLOR_BGR2RGB)
#img_pan = cv2.cvtColor(img_pan, cv2.COLOR_BGR2RGB)
#convert image to RGB

#cv2.imshow('something', img[:,:,0])
#cv2.waitKey(0)

model = cnn.Model()

def test_convolve2D(data, kernel, stride, padding):
    img = data
    temp2 = []
    convolved_output = []
    kernel_size = kernel.shape

    output_shape = cnn.Layer.calc_output_size_padMode(img.shape, np.array(kernel_size), np.array(stride), padding)

    if padding == 'full':
        img = cnn.Layer.addPadding(data, kernel_size, stride)

    temp4 = []
    for z in range(len(img.shape)):
        temp2 = img[:,:,z]
        temp3 = []

        for i in np.arange(temp2.shape[0], step=stride[0]):
            for j in np.arange(temp2.shape[1], step=stride[1]):
                temp = temp2[i:i+kernel_size[0], j:j+kernel_size[1]]  
                if temp.shape == tuple(kernel_size):
                    temp3.append((temp*kernel).sum())

        temp4.append(np.array(temp3).reshape(output_shape))

    convolved_output = np.dstack(temp4)

    return convolved_output

#Padding = Layer()
#data = Padding.addPadding(img_pan, [20,16], [10,8])

#poolingLayer = MaxPoolingLayer2D([1,1], [2,1], keepDims = True, stride_padding = [100,100], kernel_size = [50,26])
#poolingLayer = MaxPoolingLayer2D([10,18], [2,2])
#new_data = poolingLayer.maxPooling2d(img_pan)

kernel_random = np.random.randint(2, size=(50,50))/255 #random kernel
kernel_random = kernel_random / np.sum(kernel_random)
kernel_edge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) #edge detection kernel
kernel_identity = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]]) #identity kernel
kernel_sharpening = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]]) #sharpening kernel
kernel_gaussian_blur = (1/16)*np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])

#conv = ConvLayer2D(2, 3, 'valid', 1, name='Something')

#data = conv.forward(img_pan, True)

#conv2dLayer = ConvLayer2D(1, kernel.shape, 'valid', 2, 'reLU')
#conv_data = conv2dLayer.convolve2D(img_letter[:,:,0], kernel)
#conv_data2 = conv2dLayer.convolve2D(img_pan[:,:,0], kernel)

model.add(cnn.ConvLayer2D(1, kernel_edge, 'valid', 1, name='Something'))
#model.add(cnn.ReLU_activation())
model.add(cnn.ConvLayer2D(1, kernel_identity, 'valid', 1, name='Something'))
 
#model.add(cnn.ReLU_activation())
#model.add(MaxPoolingLayer2D([1,1], [2,2], keepDims = False, stride_padding = [100,100], kernel_size = [50,26]))

#conv_data = model.layers['ConvLayer2D-Something-0'].convolve2D(img_letter, kernel_identity)
#conv_data2 = model.layers['ConvLayer2D-Something-0'].convolve2D(img_pan, kernel_identity)

#cv2.imshow('something', new_data)

#cv2.imshow('original_letter', img_letter)
#cv2.imshow('conv_letter', conv_data)

#cv2.imshow('original_panorama', img_pan)
#cv2.imshow('conv_panorama', conv_data2)

start = time.time()
data = model.forward(img_pan)
end = time.time()

print(end - start)

cv2.imshow('original_panorama', img_pan)
for i in range(data.shape[0]):
    cv2.imshow(f'conv - {i}', data[i])

cv2.waitKey(0)