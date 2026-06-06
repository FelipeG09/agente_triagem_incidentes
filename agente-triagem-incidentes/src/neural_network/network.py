"""
network.py — Feedforward neural network with externally-set weights.
Rede neural feedforward com pesos definidos externamente pelo AG.
"""

import numpy as np
from .activations import tanh, softmax


class NeuralNetwork:
    """
    Two-layer feedforward neural network for incident classification.

    Rede neural feedforward de duas camadas para classificação de incidentes.
    Arquitetura: entrada → camada oculta (tanh) → saída (softmax).
    Os pesos NÃO são treinados por backpropagation — são otimizados
    externamente pelo Algoritmo Genético.

    Atributos:
        input_size: número de features de entrada
        hidden_size: número de neurônios na camada oculta
        output_size: número de classes de saída
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize network architecture with zero weights.

        Inicializa a arquitetura da rede com pesos zerados.
        Os pesos reais são definidos pelo método set_weights().
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Matrizes de pesos e biases
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def get_weights_count(self) -> int:
        """
        Return total number of trainable parameters.

        Retorna o número total de parâmetros treináveis da rede.
        Inclui todos os pesos e biases das duas camadas.
        """
        return (
            self.input_size * self.hidden_size   # W1
            + self.hidden_size                    # b1
            + self.hidden_size * self.output_size # W2
            + self.output_size                    # b2
        )

    def get_weights(self) -> np.ndarray:
        """
        Return all parameters as a flat 1D array.

        Retorna todos os parâmetros concatenados em um vetor 1D.
        Ordem: W1 (achatado), b1, W2 (achatado), b2.
        Usado pelo AG para manipular os pesos como um indivíduo.
        """
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def set_weights(self, flat_array: np.ndarray) -> None:
        """
        Reconstruct weight matrices from a flat weight vector.

        Reconstrói as matrizes de pesos a partir de um vetor plano.
        Usado pelo AG para injetar pesos otimizados na rede.

        Parâmetros:
            flat_array: vetor 1D com todos os parâmetros concatenados
        """
        idx = 0

        w1_size = self.input_size * self.hidden_size
        self.W1 = flat_array[idx: idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size

        self.b1 = flat_array[idx: idx + self.hidden_size]
        idx += self.hidden_size

        w2_size = self.hidden_size * self.output_size
        self.W2 = flat_array[idx: idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size

        self.b2 = flat_array[idx: idx + self.output_size]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run forward pass and return class probability distributions.

        Executa a passagem direta e retorna distribuições de probabilidade.
        Shape de entrada: (n_samples, input_size)
        Shape de saída: (n_samples, output_size)

        Parâmetros:
            X: matriz de features de entrada

        Retorna:
            Matriz de probabilidades por classe (soma de cada linha = 1)
        """
        # Camada oculta com tanh
        hidden = tanh(X @ self.W1 + self.b1)
        # Camada de saída com softmax
        output = softmax(hidden @ self.W2 + self.b2)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class labels (argmax of probabilities).

        Retorna os rótulos de classe preditos (argmax das probabilidades).
        Shape de entrada: (n_samples, input_size)
        Shape de saída: (n_samples,) com inteiros 0, 1 ou 2

        Parâmetros:
            X: matriz de features de entrada

        Retorna:
            Array 1D de rótulos preditos
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
