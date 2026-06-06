"""
triage_agent.py — Intelligent incident triage agent combining RNA and AG.
Agente inteligente de triagem de incidentes combinando RNA e AG.
"""

import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.neural_network.network import NeuralNetwork
from src.genetic.algorithm import GeneticAlgorithm
from config import ProjectConfig


# Mapeamento de índices para rótulos em português
LABEL_MAP_PT = {
    0: "Ignorar",
    1: "Responder",
    2: "Escalar Imediatamente",
}


class TriageAgent:
    """
    Intelligent IT incident triage agent powered by RNA + AG hybrid.

    Agente inteligente de triagem de incidentes de TI baseado em
    abordagem híbrida RNA + Algoritmo Genético.

    O agente utiliza uma Rede Neural Artificial para classificar
    tickets de incidentes em três categorias:
      0 → Ignorar (baixa prioridade)
      1 → Responder (média prioridade)
      2 → Escalar Imediatamente (alta prioridade)

    Os pesos da RNA são otimizados por um Algoritmo Genético,
    eliminando a necessidade de backpropagation.

    Atributos:
        config: configuração completa do projeto
        network: instância da RNA
        ga: instância do Algoritmo Genético
        evolution_history: histórico de fitness por geração
        best_generation: geração em que o melhor fitness foi atingido
    """

    def __init__(self, config: ProjectConfig) -> None:
        """
        Initialize the triage agent with configuration.

        Inicializa o agente de triagem com a configuração do projeto.
        Cria instâncias da RNA e do AG prontas para treinamento.
        """
        self.config = config
        self.network = NeuralNetwork(
            input_size=config.rna.input_size,
            hidden_size=config.rna.hidden_size,
            output_size=config.rna.output_size,
        )
        self.ga = GeneticAlgorithm(config, self.network)
        self.evolution_history: Dict[str, List[float]] = {}
        self.best_generation: int = 0
        self._is_trained: bool = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Train the agent by running the genetic algorithm.

        Treina o agente executando o Algoritmo Genético.
        Ao final, configura a RNA com os melhores pesos encontrados.

        Parâmetros:
            X_train: features de treino (n_samples, 8)
            y_train: rótulos de treino (n_samples,)
            X_val: features de validação
            y_val: rótulos de validação

        Retorna:
            Dicionário com histórico de fitness por geração
        """
        best_weights, history = self.ga.run(X_train, y_train, X_val, y_val)

        # Configurar RNA com os melhores pesos
        self.network.set_weights(best_weights)
        self.evolution_history = history
        self._is_trained = True

        # Identificar geração do melhor fitness
        best_fitness_values = history["best_fitness"]
        self.best_generation = int(np.argmax(best_fitness_values)) + 1

        return history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agent on the test set.

        Avalia o agente treinado no conjunto de teste.
        Calcula métricas completas de classificação.

        Parâmetros:
            X_test: features de teste
            y_test: rótulos verdadeiros de teste

        Retorna:
            Dicionário com métricas: accuracy, f1_weighted, report, confusion_matrix
        """
        predictions = self.network.predict(X_test)
        probabilities = self.network.forward(X_test)

        target_names = self.config.dataset.class_names_report
        report = classification_report(
            y_test,
            predictions,
            target_names=target_names,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, predictions)
        acc = accuracy_score(y_test, predictions)
        f1_w = f1_score(y_test, predictions, average="weighted", zero_division=0)

        return {
            "accuracy": float(acc),
            "f1_weighted": float(f1_w),
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    def predict(
        self,
        ticket_features: np.ndarray,
    ) -> Tuple[int, str, float]:
        """
        Predict triage decision for a single ticket.

        Prediz a decisão de triagem para um único ticket.

        Parâmetros:
            ticket_features: vetor 1D de 8 features normalizadas

        Retorna:
            Tupla (label_int, label_pt_br, confidence_pct) onde:
              - label_int: 0, 1 ou 2
              - label_pt_br: texto em português
              - confidence_pct: confiança em percentual (0-100)
        """
        if ticket_features.ndim == 1:
            X = ticket_features.reshape(1, -1)
        else:
            X = ticket_features

        probabilities = self.network.forward(X)[0]
        label_int = int(np.argmax(probabilities))
        label_pt = LABEL_MAP_PT[label_int]
        confidence = float(probabilities[label_int] * 100)

        return label_int, label_pt, confidence

    def save_model(self, path: str) -> None:
        """
        Save trained model weights and config to disk.

        Salva os pesos do modelo treinado e a configuração em disco.
        Usa pickle para serialização.

        Parâmetros:
            path: caminho do arquivo de saída (.pkl)
        """
        model_data = {
            "weights": self.network.get_weights(),
            "config": self.config,
            "evolution_history": self.evolution_history,
            "best_generation": self.best_generation,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str) -> None:
        """
        Load a previously saved model from disk.

        Carrega um modelo previamente salvo do disco.
        Restaura pesos e configuração a partir do arquivo pkl.

        Parâmetros:
            path: caminho do arquivo salvo (.pkl)
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.config = model_data["config"]
        self.network.set_weights(model_data["weights"])
        self.evolution_history = model_data.get("evolution_history", {})
        self.best_generation = model_data.get("best_generation", 0)
        self._is_trained = True

    def get_evolution_summary(self) -> str:
        """
        Return a formatted Portuguese summary of the evolution results.

        Retorna um resumo formatado em português dos resultados da evolução.
        Inclui estatísticas de fitness e progresso ao longo das gerações.

        Retorna:
            String formatada com o resumo da evolução
        """
        if not self.evolution_history:
            return "Modelo ainda não treinado."

        best_list = self.evolution_history["best_fitness"]
        mean_list = self.evolution_history["mean_fitness"]
        total_gen = len(best_list)
        final_best = max(best_list)
        initial_best = best_list[0]
        improvement = final_best - initial_best

        lines = [
            "=" * 60,
            "  RESUMO DA EVOLUÇÃO DO ALGORITMO GENÉTICO",
            "=" * 60,
            f"  Total de gerações: {total_gen}",
            f"  Melhor fitness inicial (geração 1): {initial_best:.4f}",
            f"  Melhor fitness final: {final_best:.4f}",
            f"  Geração do melhor resultado: {self.best_generation}",
            f"  Melhoria total: +{improvement:.4f}",
            f"  Fitness médio final: {mean_list[-1]:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
