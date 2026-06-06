"""
fitness.py — Fitness function for evaluating neural network weight vectors.
Função de fitness para avaliar vetores de pesos da rede neural.
"""

import numpy as np
from sklearn.metrics import f1_score
from src.neural_network.network import NeuralNetwork
from config import ProjectConfig


def compute_fitness(
    network: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    config: ProjectConfig,
) -> float:
    """
    Compute fitness score for a neural network configuration.

    Calcula o score de fitness para uma configuração de rede neural.

    Componentes do fitness:
    1. F1-score ponderado (weighted): mede qualidade geral da classificação
    2. Penalidade crítica: subtrai 0.15 para cada incidente crítico (label 2)
       classificado como 'ignorar' (label 0) — erro gravíssimo em operações reais

    A penalidade crítica modela a consequência real de ignorar um incidente
    de alta severidade, como uma falha de segurança não tratada.

    Parâmetros:
        network: instância da RNA com pesos já configurados
        X: matriz de features (n_samples, n_features)
        y: rótulos verdadeiros (n_samples,)
        config: configuração do projeto com valor da penalidade

    Retorna:
        float no intervalo aproximado [0, 1] — maior é melhor
    """
    predictions = network.predict(X)

    # F1-score ponderado como base do fitness
    base_fitness = f1_score(y, predictions, average="weighted", zero_division=0)

    # Penalidade: incidentes críticos (label=2) classificados como ignorar (label=0)
    critical_mask = y == 2
    if np.sum(critical_mask) > 0:
        critical_as_ignore = np.sum(
            (y == 2) & (predictions == 0)
        )
        penalty = critical_as_ignore * config.critical_penalty
    else:
        penalty = 0.0

    fitness = base_fitness - penalty
    return float(np.clip(fitness, 0.0, 1.0))
