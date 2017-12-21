from random import *
from matplotlib import pyplot as plt

MAX = 1300000

class Perceptron:
    # Data members of perceptron class
    input_X_list = []
    input_answer_list = []
    test_list = []
    weights = []
    bias = []
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    learning_rate = 0

    # Constructor to initialise the data members
    def __init__(self):
        self.weights = [randint(-1, 1), randint(-1, 1)]
        self.bias = [randint(0, 1), randint(0, 1)]
        self.learning_rate = 0.1 # Industry standard? 0.01 doesn't converge quickly
        print("Initial Weights: ", self.weights, " and bias ", self.bias)

        # Generate the train and testing list
        self.create_train_list()
        self.create_test_list()

        # Train the model
        for input_list, output in zip(self.input_X_list, self.input_answer_list):
            self.train(input_list, output)

        # Predicting values
        for test in self.test_list:
            self.predict(test)

        # Final weights and bias
        print("Final Weights: ", self.weights, " and bias ", self.bias)

        # Plot the graph for easy visualisation
        self.plot_graph()

    # Generates a list of training data consisting of features and labels
    def create_train_list(self):
        for i in range(0, 1000000):
            x = randint(-1000000, 1000000)
            y = randint(-1000000, 1000000)
            self.input_X_list += [[x, y]]

            if(y > self.polynomial(x)):
                output = 1
            else:
                output = -1
            self.input_answer_list += [output]

    # Generates a list of testing data
    def create_test_list(self):
        for i in range(0, 1000):
            x = randint(-2500000, 2500000)
            y = randint(-2500000, 2500000)
            inp = [x, y]
            self.test_list += [inp]

    # includes feed forward and back propagation
    def train(self, input_var, output):
        v = 0
        print("TRAINING")
        for i in range(len(self.weights)):
            v += (self.weights[i] * input_var[i])  # + bias[i]
        predicted = self.activate(v)
        error = output - predicted
        change_in_weight = error * self.learning_rate
        # print(output, " ", predicted, " ", change_in_weight)
        for i in range(len(self.weights)):
            self.weights[i] += change_in_weight * input_var[i]

    # predicts the value of test data
    def predict(self, test_var):
        # print("PREDICTING\n")
        v = 0
        for i in range(len(self.weights)):
            v += (self.weights[i] * test_var[i])  # + bias[i]
        output = self.activate(v)
        if(output == 1):
            self.X1 += [test_var[0]]
            self.Y1 += [test_var[1]]
        else:
            self.X2 += [test_var[0]]
            self.Y2 += [test_var[1]]

    # a simple activation function
    def activate(self, n):
        if(n > 0):
            return 1
        else:
            return -1

    # a simple polynomial function for classifition purposes
    def polynomial(self, x):
        y = 2*x + 5
        return y

    # Plot the graph
    def plot_graph(self):
        plt.scatter(self.X1, self.Y1, label='1', color='b', s=50)
        plt.scatter(self.X2, self.Y2, label='-1', color = 'r', s=50)
        plt.plot([-MAX, 0, MAX], [self.polynomial(-MAX), self.polynomial(0), self.polynomial(MAX)], color='k')
        plt.show()

# create the perceptron
p = Perceptron()
