from model import FFNeuralNetwork
import numpy as np

# MAIN
dataset = np.array([[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]])

nb_hidden_layer = 1
nb_nodes = 2
nb_input = dataset.shape[1] - 1
FFNN = FFNeuralNetwork(nb_hidden_layer, nb_nodes, nb_input)

epoch = 10
l_rate = 0.001
momentum = 0.25
batch_size = 2

FFNN.fit(dataset, epoch, l_rate, momentum, batch_size)
print(FFNN.predict(dataset[:,:-1]))
