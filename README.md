# messy-project-
hmmmmmmmmm
import numpy as np
import sys
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000,28*28) / 255 , y_train[0:1000])
one_hot = np.zeros((len(labels),10))
for i,l in enumerate(labels):
    one_hot[i][l] = 1
labels = one_hot
test_image = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1
np.random.seed(1)
def relu(x):
    return np.maximum(0,x)
def derivative_relu(x):
    return (x > 0).astype(float)
learning_rate, iterations, hidden_size, inputs, outputs = \
 (0.005, 350, 40, 784, 10)
weights_hidden = 0.2 * np.random.random((inputs,hidden_size)) - 0.1
weights_output = 0.2 * np.random.random((hidden_size,outputs)) - 0.1
for n in range(iterations):
    error,correct = (0.0,0)
    for i in range(len(images)):
        input  = images[i:i+1]
        hidden_z= np.dot(input,weights_hidden)
        hidden_relu = relu(hidden_z)
        drop_out = np.random.randint(2,size=hidden_relu.shape)
        hidden_relu *= drop_out * 2
        output_z = np.dot(hidden_relu,weights_output)
        output_relu = relu(output_z)
        error += np.sum((labels[i:i+1] - output_relu)**2 )
        correct += int(np.argmax(output_relu) == np.argmax(labels[i:i+1]))
        gradient_aoutput  = output_relu - labels[i:i+1]
        gradient_zoutput = gradient_aoutput * derivative_relu(output_z)
        weights_output -= learning_rate * np.dot(hidden_relu.T,gradient_zoutput)
        gradient_ahidden  = np.dot(gradient_zoutput,weights_output.T)
        gradient_zhidden = gradient_ahidden * derivative_relu(hidden_z)
        gradient_zhidden *= drop_out
        weights_hidden -= learning_rate * np.dot(input.T,gradient_zhidden)

    if(n%10 == 0):
      test_error = 0.0
      test_correct = 0
      for i in range(len(test_image)):

          input = test_image[i:i+1]
          hidden = relu(np.dot(input,weights_hidden))
          output = relu(np.dot(hidden,weights_output))
          test_error += np.sum((test_labels[i:i+1]- output) ** 2)
          test_correct += int(np.argmax(output) == np.argmax(test_labels[i]))


     sys.stdout.write(
          f"\nI{n}"
          f"Test-error : {(test_error / len(test_labels)):.3f}"
          f"Test-accuracy : {(test_correct / len(test_labels)):.3f}"
          f"Training error : {(error / len(labels)):.3f}"
          f"Training accuracy : {(correct / len(labels)):.3f}"
    )




