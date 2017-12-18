from random import *


class Perceptron:
    input_X_list = [[1, 2]]
    input_answer_list = []
    output = []
    weights = []
    bias = []
    learning_rate = 0

    def __init__(self):
        self.weights = [randint(-1, 1), randint(-1, 1)]
        self.bias = [randint(0, 1), randint(0, 1)]
        learning_rate = 0.01
        print("Initial Weights: ", self.weights, " and bias ", self.bias)

    def train(self, input_var, output):
        v = 0
        print("TRAINING")
        for i in range(len(self.weights)):
            v += (self.weights[i] * input_var[i])  # + bias[i]
        predicted = self.activate(v)
        error = output - predicted
        change_in_weight = error * self.learning_rate
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + \
                (change_in_weight * input_var[i])

    def predict(self, test_var):
        print("PREDICTING\n")
        v = 0
        for i in range(len(self.weights)):
            v += (self.weights[i] * input_var[i])  # + bias[i]
        output = self.activate(v)

    def activate(self, n):
        if(n > 0):
            return 1
        else:
            return -1

    def linear_fn(x):
        y = 0.1*x + 0.5
        return y

p = Perceptron()
