"""
config.py — Central configuration for the incident triage agent project.
Configuração central do projeto agente de triagem de incidentes.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RNAConfig:
    """
    Neural network architecture configuration.

    Configuração da arquitetura da Rede Neural Artificial (RNA).
    Define os tamanhos das camadas e funções de ativação utilizadas.
    """
    input_size: int = 8
    hidden_size: int = 12
    output_size: int = 3
    hidden_activation: str = "tanh"
    output_activation: str = "softmax"


@dataclass
class AGConfig:
    """
    Genetic algorithm hyperparameters.

    Hiperparâmetros do Algoritmo Genético (AG).
    Controla o comportamento da evolução da população.
    """
    population_size: int = 100
    generations: int = 150
    mutation_rate: float = 0.08
    mutation_strength: float = 0.2
    crossover_rate: float = 0.75
    tournament_size: int = 5
    elitism_count: int = 5


@dataclass
class DatasetConfig:
    """
    Dataset generation configuration.

    Configuração para geração do conjunto de dados sintético.
    Define distribuição de classes e parâmetros de divisão treino/teste.
    """
    n_samples: int = 1000
    class_distribution: List[float] = field(
        default_factory=lambda: [0.65, 0.25, 0.10]
    )
    random_seed: int = 42
    train_split: float = 0.8

    # Rótulos das classes em português
    class_labels_pt: List[str] = field(
        default_factory=lambda: ["Ignorar", "Responder", "Escalar Imediatamente"]
    )
    class_names_report: List[str] = field(
        default_factory=lambda: ["ignorar", "responder", "escalar"]
    )


@dataclass
class ProjectConfig:
    """
    Top-level project configuration combining all sub-configs.

    Configuração principal do projeto, agrupando todas as sub-configurações.
    Ponto único de acesso a todos os hiperparâmetros do sistema.
    """
    rna: RNAConfig = field(default_factory=RNAConfig)
    ag: AGConfig = field(default_factory=AGConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Caminhos de saída
    results_dir: str = "results"
    reports_dir: str = "reports"
    model_path: str = "results/best_model.pkl"
    chart_path: str = "results/evolucao_fitness.png"
    report_path: str = "reports/relatorio_final.txt"

    # Penalidade para erros críticos (escalar classificado como ignorar)
    critical_penalty: float = 0.15


def get_config() -> ProjectConfig:
    """
    Returns the default project configuration.

    Retorna a configuração padrão do projeto com todos os hiperparâmetros.
    """
    return ProjectConfig()
