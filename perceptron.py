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


p = Perceptron()
