import cPickle

import mnist_loader
import network
import numpy as np
from scipy import misc, float32
import glob
training_data1, validation_data, test_data = mnist_loader.load_data_wrapper()
#print training_data[0][0]
training_data = []
folders = glob.glob("C:/Users/ajha2/Desktop/PythonProjects/MNIST_image/mnist_png/testing/*")

for folder in folders:
    files = glob.glob( folder + "/*")
    for file in files:
        img = misc.imread(file)
        img_pixel_array = []
        for i in range(0, len(img)):
            for j in range(0, len(img[0])):
                img_pixel_array.append([float32(img[i][j])/256])
        img_pixel_np_array = np.array(img_pixel_array, ndmin = 2)
        #img_digit = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], ndmin = 2)
        #img_digit[int(folder[-1])][0] = 1.0
        training_data.append((img_pixel_np_array, int(folder[-1])))

f = open('C:/Users/ajha2/Desktop/PythonProjects/MNIST_image/testing_image_objects.save', 'wb')

cPickle.dump(training_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
print
