from random import *


class Perceptron:
    input_X_list = []
    input_answer_list = []
    test_list = []
    weights = []
    bias = []
    learning_rate = 0

    def __init__(self):
        self.weights = [randint(-1, 1), randint(-1, 1)]
        self.bias = [randint(0, 1), randint(0, 1)]
        learning_rate = 0.01
        print("Initial Weights: ", self.weights, " and bias ", self.bias)

        self.create_train_list()
        self.create_test_list()
        for input_list, output in zip(self.input_X_list, self.input_answer_list):
            self.train(input_list, output)
        for test in self.test_list:
            self.predict(test)

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

    def create_test_list(self):
        for i in range(0, 100):
            x = randint(-500000, 500000)
            y = randint(-500000, 500000)
            inp = [x, y]
            self.test_list += [inp]

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
            v += (self.weights[i] * test_var[i])  # + bias[i]
        output = self.activate(v)

    def activate(self, n):
        if(n > 0):
            return 1
        else:
            return -1

    def polynomial(self, x):
        y = 2*x + 5
        return y

p = Perceptron()
