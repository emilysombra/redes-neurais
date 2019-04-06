from random import random
from active import signal
import numpy as np


class Perceptron:
    def __init__(self, lr=0.03, func=signal):
        self.__learn_ratio = lr  # Taxa de Aprendizado (padrão: 0.03)
        self.__weights = []  # Lista de Pesos
        self.__activation = func  # Função de ativação (padrão: signal)
        self.epocas = -1  # Contador de épocas

    def __add_bias(self, v):
        new = []
        for i in v:
            new.append(np.insert(i, 0, -1))
        return new

    def fit(self, stdX, D):
        # Adicionando entrada x0 = -1 (Valor padrão para o limiar)
        # cada linha Xi de X := [-1] + stdX[i]
        X = self.__add_bias(stdX)
        # Pesos iniciados aleatoriamente
        # Qtd de pesos == qtd de elementos em cada xi em X
        for i in range(len(X[0])):
            self.__weights.append(random())

        # Contador de épocas = 0
        self.epocas = 0

        while(True):
            # Flag para indicar se há erro no neurônio
            erro = False

            # Percorre todas as entradas
            for i in range(len(X)):
                xk = X[i]
                dk = D[i]
                # Produto interno de xk e w (weights)
                # Pode ser visto como um somatorio de wi * xi para todo i
                u = float(np.dot(xk, self.__weights))
                # Função de ativação (padrão: y = signal(u))
                y = self.__activation(u)
                # Se houve um erro, i.e., y != dk:
                if(y != dk):
                    # Sinaliza o erro
                    erro = True
                    # Constante K que será multiplicado por xk
                    K = self.__learn_ratio * (dk - y)
                    # Vetor v = K * xk
                    v = np.multiply(K, xk)
                    # Ajusta os pesos com base em v
                    self.__weights = np.add(self.__weights, v)

            # Incrementa o contador de épocas
            self.epocas += 1
            # Caso não haja erros, para
            if(not erro):
                break

    def predict(self, X):
        # Lista de previsoes
        previsoes = []
        X = self.__add_bias(X)
        for xk in X:
            # Produto interno de x e w (weights)
            # Pode ser visto como um somatorio de wi * xi para todo i
            u = float(np.dot(xk, self.__weights))
            # Função de ativação (padrão: y = signal(u))
            y = self.__activation(u)
            # Adiciona y à lista de previsões
            previsoes.append(y)

        # Retorna a lista de previsões
        return previsoes


if(__name__ == '__main__'):
    p = Perceptron()
