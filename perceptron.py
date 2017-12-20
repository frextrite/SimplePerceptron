from random import *


class Perceptron:
    # Data members of perceptron class
    input_X_list = []
    input_answer_list = []
    test_list = []
    weights = []
    bias = []
    learning_rate = 0

    # Constructor to initialise the data members
    def __init__(self):
        self.weights = [randint(-1, 1), randint(-1, 1)]
        self.bias = [randint(0, 1), randint(0, 1)]
        learning_rate = 0.01 # Industry standard?
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

    # Generates a list of training data consisting of features and labels
    def create_train_list(self):
        for i in range(0, 10000):
            x = randint(-500000, 500000)
            y = randint(-500000, 500000)
            self.input_X_list += [[x, y]]

            if(y > self.polynomial(x)):
                output = 1
            else:
                output = -1
            self.input_answer_list += [output]

    # Generates a list of testing data
    def create_test_list(self):
        for i in range(0, 100):
            x = randint(-500000, 500000)
            y = randint(-500000, 500000)
            inp = [x, y]
            self.test_list += [inp]

    # includes feed forward and back propagation
    def train(self, input_var, output):
        v = 0
        # print("TRAINING")
        for i in range(len(self.weights)):
            v += (self.weights[i] * input_var[i])  # + bias[i]
        predicted = self.activate(v)
        error = output - predicted
        change_in_weight = error * self.learning_rate
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + \
                (change_in_weight * input_var[i])

    # predicts the value of test data
    def predict(self, test_var):
        # print("PREDICTING\n")
        v = 0
        for i in range(len(self.weights)):
            v += (self.weights[i] * test_var[i])  # + bias[i]
        output = self.activate(v)

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

# create the perceptron
p = Perceptron()

# TODO: create a graph for easy interpretation
