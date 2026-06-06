"""
algorithm.py — Genetic Algorithm for optimizing neural network weights.
Algoritmo Genético para otimização dos pesos da rede neural.
"""

import numpy as np
from typing import List, Dict, Tuple, Any

from src.neural_network.network import NeuralNetwork
from config import ProjectConfig
from .fitness import compute_fitness
from .operators import uniform_crossover, gaussian_mutation
from .selection import tournament_selection


class GeneticAlgorithm:
    """
    Evolutionary optimizer using a Genetic Algorithm for RNA weight optimization.

    Otimizador evolucionário que usa Algoritmo Genético para otimização
    dos pesos da RNA. A cada geração:
    1. Avalia o fitness de todos os indivíduos
    2. Preserva os melhores (elitismo)
    3. Seleciona pais por torneio
    4. Gera filhos por crossover uniforme
    5. Aplica mutação gaussiana
    6. Repete até o número máximo de gerações

    Atributos:
        config: configuração completa do projeto
        network_template: instância de RNA usada como modelo de arquitetura
    """

    def __init__(self, config: ProjectConfig, network_template: NeuralNetwork) -> None:
        """
        Initialize the Genetic Algorithm with configuration and network template.

        Inicializa o Algoritmo Genético com configuração e template de rede.
        O network_template define apenas a arquitetura — seus pesos são
        substituídos durante a evolução.
        """
        self.config = config
        self.network = network_template
        self.n_weights = network_template.get_weights_count()
        self._best_ever_weights: np.ndarray = None
        self._best_ever_fitness: float = -np.inf

    def _initialize_population(self) -> List[np.ndarray]:
        """
        Create initial population with random Gaussian weight vectors.

        Cria a população inicial com vetores de pesos gaussianos aleatórios.
        Desvio padrão de 0.5 fornece boa cobertura do espaço de busca inicial.

        Retorna:
            Lista de vetores de pesos (população inicial)
        """
        np.random.seed(self.config.dataset.random_seed)
        population = [
            np.random.normal(0, 0.5, self.n_weights)
            for _ in range(self.config.ag.population_size)
        ]
        return population

    def _evaluate_population(
        self,
        population: List[np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[float]:
        """
        Evaluate fitness for all individuals in the population.

        Avalia o fitness de todos os indivíduos da população.
        Cada indivíduo (vetor de pesos) é injetado na rede e avaliado.

        Retorna:
            Lista de valores de fitness para cada indivíduo
        """
        scores = []
        for weights in population:
            self.network.set_weights(weights)
            score = compute_fitness(self.network, X, y, self.config)
            scores.append(score)
        return scores

    def _apply_elitism(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
    ) -> List[np.ndarray]:
        """
        Select the top N individuals for elitism carry-over.

        Seleciona os N melhores indivíduos para preservação por elitismo.
        Estes indivíduos passam intactos para a próxima geração, garantindo
        que o melhor fitness nunca diminua ao longo das gerações.
        """
        n_elite = self.config.ag.elitism_count
        sorted_idx = np.argsort(fitness_scores)[::-1]
        return [population[i].copy() for i in sorted_idx[:n_elite]]

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        """
        Run the full evolutionary optimization loop.

        Executa o loop completo de otimização evolucionária.
        Retorna os melhores pesos encontrados em qualquer geração e o
        histórico de fitness para plotagem da curva de evolução.

        Parâmetros:
            X_train: features de treino
            y_train: rótulos de treino
            X_val: features de validação
            y_val: rótulos de validação

        Retorna:
            (best_weights, history_dict) onde history_dict contém
            listas 'best_fitness', 'mean_fitness', 'std_fitness'
        """
        population = self._initialize_population()
        history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
            "std_fitness": [],
        }

        ag_cfg = self.config.ag

        for generation in range(ag_cfg.generations):
            # Avaliar toda a população nos dados de treino
            fitness_scores = self._evaluate_population(population, X_train, y_train)
            fitness_arr = np.array(fitness_scores)

            # Registrar estatísticas
            gen_best = float(np.max(fitness_arr))
            gen_mean = float(np.mean(fitness_arr))
            gen_std = float(np.std(fitness_arr))

            history["best_fitness"].append(gen_best)
            history["mean_fitness"].append(gen_mean)
            history["std_fitness"].append(gen_std)

            # Atualizar melhor global encontrado em qualquer geração
            if gen_best > self._best_ever_fitness:
                self._best_ever_fitness = gen_best
                best_idx = int(np.argmax(fitness_arr))
                self._best_ever_weights = population[best_idx].copy()

            # Log a cada 10 gerações
            if (generation + 1) % 10 == 0 or generation == 0:
                self._log_generation(generation + 1, gen_best, gen_mean)

            # Construir próxima geração
            elite = self._apply_elitism(population, fitness_scores)
            n_offspring = ag_cfg.population_size - len(elite)

            # Seleção de pais por torneio
            parents = tournament_selection(
                population,
                fitness_scores,
                ag_cfg.tournament_size,
                n_offspring * 2,
            )

            # Gerar filhos via crossover + mutação
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child = uniform_crossover(parents[i], parents[i + 1], ag_cfg.crossover_rate)
                child = gaussian_mutation(child, ag_cfg.mutation_rate, ag_cfg.mutation_strength)
                offspring.append(child)
                if len(offspring) >= n_offspring:
                    break

            # Completar se necessário
            while len(offspring) < n_offspring:
                offspring.append(parents[len(offspring) % len(parents)].copy())

            population = elite + offspring[:n_offspring]

        return self._best_ever_weights, history

    def _log_generation(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
    ) -> None:
        """
        Print generation statistics to console.

        Imprime estatísticas da geração no console.
        Chamado a cada 10 gerações durante a evolução.
        """
        print(
            f"  Geração {generation:4d}/{self.config.ag.generations} | "
            f"Melhor Fitness: {best_fitness:.4f} | "
            f"Fitness Médio: {mean_fitness:.4f}"
        )
