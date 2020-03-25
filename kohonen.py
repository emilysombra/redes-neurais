from random import random
from scipy.spatial import distance
import numpy as np
import math


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
            is_done = False
            # para cada entrada xi no conjunto de entradas, faça:
            for xi in X:
                # normalizando entrada
                xi = normalize(xi)
                # variaveis para indicar o 'neuronio' mais próximo e a distancia
                nearest = None
                lowest_distance = float('inf')
                neighbors = []
                # checando a distancia entre xi
                n = len(self.__neurons)
                raiz_n = int(math.sqrt(n))
                for i in range(n):
                    neuron = self.__neurons[i]
                    current_distance = neuron.distance_to_input(xi)
                    if current_distance < lowest_distance:
                        lowest_distance = current_distance
                        nearest = neuron
                        neighbors = []
                        if i - 1 > -1:
                            neighbors.append(i - 1)
                        if i - raiz_n > -1:
                            neighbors.append(i - 1)
                        if i + 1 < n:
                            neighbors.append(i - 1)
                        if i + raiz_n < n:
                            neighbors.append(i - 1)
                
                # ajuste de pesos
                ajuste = (self.__learn_ratio * lowest_distance)
                if ajuste < 0.0001:
                    is_done = True
                nearest.weights = nearest.weights + ajuste
                # ajuste de pesos dos vizinhos
                for n in neighbors:
                    n = self.__neurons[n]
                    ajuste = ((self.__learn_ratio / 2) * lowest_distance)
                    n.weights = n.weights + ajuste
            epochs += 1
            if is_done:
                break

    def predict(self, X):
        respostas = []
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
            respostas.append(nearest)
        return respostas
