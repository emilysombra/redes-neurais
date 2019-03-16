from random import random


class Perceptron:
    def __init__(self, lr=0.03):
        self.learn_ratio = lr
        self.inputs = [-1]
        self.weights = [random()]

    def fit(self, x, y):
        pass

    def predict(self, x, y):
        pass

    def signal(self):
        pass


if(__name__ == '__main__'):
    p = Perceptron()
