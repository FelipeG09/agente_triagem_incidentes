"""
operators.py — Genetic operators: crossover and mutation.
Operadores genéticos: cruzamento (crossover) e mutação.
"""

import numpy as np


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Perform uniform crossover between two parent weight vectors.

    Realiza cruzamento uniforme entre dois vetores de pesos parentais.

    No cruzamento uniforme, cada gene (peso) do filho é escolhido
    independentemente de um dos pais com probabilidade igual à taxa
    de cruzamento. Preserva diversidade genética melhor que cruzamentos
    de ponto único.

    Parâmetros:
        parent1: vetor de pesos do primeiro pai
        parent2: vetor de pesos do segundo pai
        rate: probabilidade de herdar gene do parent2 (vs parent1)

    Retorna:
        Novo vetor de pesos filho combinando genes dos dois pais
    """
    mask = np.random.random(len(parent1)) < rate
    child = np.where(mask, parent2, parent1)
    return child.copy()


def gaussian_mutation(
    individual: np.ndarray,
    rate: float,
    strength: float,
) -> np.ndarray:
    """
    Apply Gaussian mutation to a weight vector.

    Aplica mutação gaussiana a um vetor de pesos.

    Cada gene é mutado independentemente com probabilidade `rate`.
    Quando mutado, o gene recebe um ruído gaussiano com desvio padrão
    `strength`. A mutação introduz variação na população, permitindo
    exploração de novas regiões do espaço de soluções.

    Parâmetros:
        individual: vetor de pesos a ser mutado
        rate: probabilidade de mutação por gene
        strength: desvio padrão do ruído gaussiano adicionado

    Retorna:
        Cópia do vetor com mutações aplicadas
    """
    mutated = individual.copy()
    mutation_mask = np.random.random(len(mutated)) < rate
    noise = np.random.normal(0, strength, len(mutated))
    mutated[mutation_mask] += noise[mutation_mask]
    return mutated
