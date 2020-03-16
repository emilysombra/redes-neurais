from random import random
from scipy.spatial import distance


class Neuron:
    def __init__(self):
        self.__weights = []

    def __str__(self):
        return str(self.__weights)
    
    def init_weights(self, size):
        for i in range(size):
            self.__weights.append(random())

    def distance_to_input(self, input):
        return distance.euclidean(self.__weights, input)


class Kohonen:
    def __init__(self, n, lr=0.03):
        self.__learn_ratio = lr  # Taxa de Aprendizado (padrão: 0.03)
        self.__neurons = []  # vetor de neurônios
        for i in range(n):
            self.__neurons[i] = Neuron()

    def fit(self, X, max_epochs=5000):
        # inicializando neuronios com valores aleatorios
        for neuron in self.__neurons:
            neuron.init_weights(len(X[0]))
        
        # Contador de épocas = 0
        epochs = 0
        while epochs < max_epochs:
            epochs += 1

    def predict(self, x):
        pass
