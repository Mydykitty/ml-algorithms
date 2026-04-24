import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from multilayer_perceptron import MultilayerPerceptron

data = pd.read_csv('../data/mnist-demo.csv')
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
# plt.figure(figsize=(10,10))
# for plot_index in range(numbers_to_display):
#     digit = data[plot_index:plot_index+1].values
#     digit_label = digit[0][0]
#     digit_pixels = digit[0][1:]
#     image_size = int(math.sqrt(digit_pixels.shape[0]))
#     frame = digit_pixels.reshape((image_size,image_size))
#     plt.subplot(num_cells,num_cells,plot_index+1)
#     plt.imshow(frame,cmap='Greys')
#     plt.title(digit_label)
# plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()

train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index).values
train_data = train_data.values

num_training_examples = 1700

x_train = train_data[:num_training_examples,1:]
y_train = train_data[:num_training_examples,[0]]

x_test = test_data[:,1:]
y_test = test_data[:,0]

layers=[784,25,10]

normalize_data = True
max_iterations = 300
alpha = 0.1

multilayer_perceptron = MultilayerPerceptron(x_train,y_train,layers,normalize_data)
multilayer_perceptron.train(max_iterations,alpha)