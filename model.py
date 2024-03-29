import numpy as np

class FFNeuralNetwork:
    def __init__(self, nb_hidden_layer, nb_nodes, nb_input):
        self.nb_hidden_layer = nb_hidden_layer
        self.nb_nodes = nb_nodes
        self.nb_input = nb_input
        self.bias = np.ones((nb_nodes, 1))
        self.init_weights()
        self.init_grad_weights()

    def init_weights(self):
        self.weights = [np.random.rand(self.nb_nodes, self.nb_input)]
        for i in range(1, self.nb_hidden_layer):
            self.weights.append(np.random.rand(self.nb_nodes, self.nb_nodes))
        self.weights.append(np.random.rand(1, self.nb_nodes))
    
    def init_grad_weights(self):
        self.grad_weights = [np.zeros((self.nb_nodes, self.nb_input))]
        for i in range(1, self.nb_hidden_layer):
            self.grad_weights.append(np.zeros((self.nb_nodes, self.nb_nodes)))
        self.grad_weights.append(np.zeros((1, self.nb_nodes)))

    def init_delta_weights(self):
        delta_weights = [np.zeros((self.nb_nodes, self.nb_input))]
        for i in range(1, self.nb_hidden_layer):
            delta_weights.append(np.zeros((self.nb_nodes, self.nb_nodes)))
        delta_weights.append(np.zeros((1, self.nb_nodes)))
        return delta_weights

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def derivative(self, output):
        return output * (1.0 - output)

    def get_weights(self):
        return self.weights

    def get_grad_weights(self):
        return self.grad_weights

    def feed_forward(self, init):
        # transpos matrix init
        output = init.reshape(init.shape[0], 1)

        for i in range(len(self.weights)):
            result = np.dot(self.weights[i], output) + 1
            output = self.sigmoid(result)
            self.output.append(output)

        return output

    # Y = Y asli satu biji
    def backward(self, Y, Y_pred):
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                error = np.sum(self.mean_square_error(Y, Y_pred), axis = 0)
            else:
                error = np.sum(self.grad_weights[i + 1] * self.weights[i], axis = 0)
            self.grad_weights[i] = error * self.derivative(self.output[i])

    # delta_weights array of matriks diinit 0 pas awal banget
    def update_weight(self, l_rate, delta_weights, momentum):
        prev_delta_weights = delta_weights
        for i in range(len(self.weights)):
            delta_weights[i] = l_rate * self.grad_weights[i] + momentum * prev_delta_weights[i]

        for weight, delta_weight in zip(self.weights, delta_weights):
            weight += delta_weight

    def mean_square_error(self, Y, Y_pred):
        return np.square(Y - Y_pred).mean()

    def create_minibatches(self, data, batch_size):
        minibatches = []
        # data = np.hstack((X, Y))
        np.random.shuffle(data) 
        n_minibatches = data.shape[0] // batch_size
    
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            minibatches.append((X_mini, Y_mini)) 
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            minibatches.append((X_mini, Y_mini)) 
        return minibatches 

    def fit(self, data, epoch, l_rate, momentum, batch_size):
        for i in range(epoch):
            print("epoch = ", i)
            print("weights = ", self.weights)
            minibatches = self.create_minibatches(data, batch_size)
            delta_weights = self.init_delta_weights()

            for minibatch in minibatches:
                X_mini, Y_mini = minibatch

                for X_train, Y_expected in zip(X_mini, Y_mini):
                    self.output = []

                    Y_pred = self.feed_forward(X_train)
                    self.backward(Y_expected, Y_pred)

                self.update_weight(l_rate, delta_weights, momentum)
             
    def predict(self, X):
        Y_pred = []
        for x in X:
            Y_pred.append(self.feed_forward(x))
        return np.array(Y_pred)
