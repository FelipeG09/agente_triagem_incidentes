"""
activations.py — Pure numpy activation functions for the neural network.
Funções de ativação implementadas com numpy puro para a rede neural.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation.

    Função de ativação ReLU (Rectificador Linear).
    Retorna max(0, x) elemento a elemento.
    """
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU function.

    Derivada da função ReLU.
    Retorna 1 onde x > 0, e 0 caso contrário.
    """
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation.

    Função de ativação tangente hiperbólica.
    Mapeia a entrada para o intervalo (-1, 1).
    Preferida na camada oculta por ser centrada em zero.
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the tanh function.

    Derivada da função tanh.
    Calculada como 1 - tanh(x)^2.
    """
    return 1.0 - np.tanh(x) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax activation.

    Função softmax numericamente estável para a camada de saída.
    Subtrai o máximo de cada linha para evitar overflow.
    Converte logits em distribuição de probabilidade (soma = 1).

    Parâmetros:
        x: array 2D de shape (n_samples, n_classes) ou 1D (n_classes,)

    Retorna:
        Array de probabilidades com mesma shape da entrada.
    """
    if x.ndim == 1:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    # 2D: subtrair máximo por linha para estabilidade numérica
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x_shifted)
    return e / np.sum(e, axis=1, keepdims=True)
