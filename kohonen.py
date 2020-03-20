from random import random
from scipy.spatial import distance
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Neuron:
    def __init__(self):
        self.weights = []

    def __str__(self):
        return str(self.weights)
    
    def init_weights(self, size):
        for i in range(size):
            self.weights.append(random())
        self.weights = normalize(self.weights)

    def distance_to_input(self, input):
        return distance.euclidean(self.weights, input)


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
            # para cada entrada xi no conjunto de entradas, faça:
            for xi in X:
                # normalizando entrada
                xi = normalize(xi)
                # variaveis para indicar o 'neuronio' mais próximo e a distancia
                nearest = None
                lowest_distance = float('inf')
                # checando a distancia entre xi
                for neuron in self.__neurons:
                    current_distance = neuron.distance_to_input(xi)
                    if current_distance < lowest_distance:
                        lowest_distance = current_distance
                        nearest = neuron
                
                # ajuste de pesos
                ajuste = (self.__learn_ratio * lowest_distance)
                nearest.weights = nearest.weights + ajuste
            epochs += 1

    def predict(self, x):
        pass
