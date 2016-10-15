import cPickle
import numpy as np
import mnist_loader
import gzip
import network
import network2
f = open('C:/Users/ajha2/Desktop/PythonProjects/MNIST_image/training_image_objects.save', 'rb')
tr_d = cPickle.load(f)

f.close()
f = open('C:/Users/ajha2/Desktop/PythonProjects/MNIST_image/testing_image_objects.save', 'rb')
te_d = cPickle.load(f)
f.close()

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#training_inputs1 = [np.reshape(x, (784, 1)) for x in training_data[0]]

#training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
#training_results = [vectorized_result(y) for y in tr_d[1]]
print "i m here"
net = network2.Network([784, 40, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(tr_d, 30, 10, 0.1, evaluation_data=te_d, lmbda=5.0, monitor_evaluation_accuracy=True, monitor_training_accuracy=True)
net.save('C:/Users/ajha2/Desktop/PythonProjects/MNIST_image/NN.json')