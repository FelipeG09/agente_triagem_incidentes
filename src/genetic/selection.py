"""
selection.py — Tournament selection for the genetic algorithm.
Seleção por torneio para o algoritmo genético.
"""

import numpy as np
from typing import List


def tournament_selection(
    population: List[np.ndarray],
    fitness_scores: List[float],
    tournament_size: int,
    n_winners: int,
) -> List[np.ndarray]:
    """
    Select individuals via tournament selection.

    Seleciona indivíduos via seleção por torneio.

    A seleção por torneio funciona da seguinte forma:
    1. Escolhe aleatoriamente `tournament_size` indivíduos da população
    2. O indivíduo com maior fitness vence o torneio
    3. O processo é repetido até selecionar `n_winners` indivíduos

    Vantagens sobre a roleta (roulette wheel):
    - Não é afetada por escalas de fitness absolutas
    - Pressão seletiva controlável pelo tamanho do torneio
    - Mais eficiente computacionalmente

    Parâmetros:
        population: lista de vetores de pesos (indivíduos)
        fitness_scores: lista de valores de fitness correspondentes
        tournament_size: número de participantes por torneio
        n_winners: número de indivíduos a selecionar

    Retorna:
        Lista de `n_winners` indivíduos selecionados (cópias)
    """
    fitness_arr = np.array(fitness_scores)
    pop_size = len(population)
    selected = []

    for _ in range(n_winners):
        # Sortear participantes do torneio sem repetição (se possível)
        tournament_size_actual = min(tournament_size, pop_size)
        competitors_idx = np.random.choice(
            pop_size,
            size=tournament_size_actual,
            replace=False,
        )
        # Vencedor: maior fitness entre os competidores
        winner_idx = competitors_idx[np.argmax(fitness_arr[competitors_idx])]
        selected.append(population[winner_idx].copy())

    return selected
