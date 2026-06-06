# src/genetic/__init__.py
from .algorithm import GeneticAlgorithm
from .fitness import compute_fitness
from .operators import uniform_crossover, gaussian_mutation
from .selection import tournament_selection
