"""
dataset.py — Synthetic IT incident ticket dataset generator.
Gerador de conjunto de dados sintético de tickets de incidentes de TI.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from config import DatasetConfig


# Nomes das features para relatórios
FEATURE_NAMES_PT = [
    "Severidade",
    "Hora Normalizada",
    "Criticidade do Sistema",
    "Palavra-chave de Segurança",
    "Palavra-chave de Infraestrutura",
    "Frequência do Remetente",
    "Recorrência",
    "Risco de SLA",
]

FEATURE_NAMES_EN = [
    "severity",
    "hour_normalized",
    "system_criticality",
    "security_keyword",
    "infra_keyword",
    "sender_frequency",
    "recurrence",
    "response_time_sla",
]


def _generate_features(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate raw feature matrix for synthetic tickets.

    Gera a matriz de features brutas para os tickets sintéticos.
    Todas as features são normalizadas entre 0.0 e 1.0.
    """
    severity = rng.beta(2, 5, n_samples)
    hour_normalized = rng.uniform(0.0, 1.0, n_samples)
    system_criticality = rng.choice([0.1, 0.5, 1.0], n_samples, p=[0.4, 0.35, 0.25])
    security_keyword = rng.choice([0.0, 1.0], n_samples, p=[0.75, 0.25])
    infra_keyword = rng.choice([0.0, 1.0], n_samples, p=[0.6, 0.4])
    sender_frequency = rng.beta(1.5, 4, n_samples)
    recurrence = rng.choice([0.0, 1.0], n_samples, p=[0.7, 0.3])
    response_time_sla = rng.beta(2, 3, n_samples)

    return np.column_stack([
        severity,
        hour_normalized,
        system_criticality,
        security_keyword,
        infra_keyword,
        sender_frequency,
        recurrence,
        response_time_sla,
    ])


def _assign_labels(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Assign labels with realistic correlations to features.

    Atribui rótulos com correlações realistas às features.
    Regras de negócio:
      - Alta severidade + keyword de segurança + fora do horário → escalar (2)
      - Recorrência alta + crítico → responder (1) ou escalar (2)
      - Baixa severidade + horário comercial → ignorar (0)
    """
    n = len(features)
    labels = np.zeros(n, dtype=int)

    severity = features[:, 0]
    hour = features[:, 1]
    system_crit = features[:, 2]
    security_kw = features[:, 3]
    infra_kw = features[:, 4]
    recurrence = features[:, 6]
    sla_risk = features[:, 7]

    # Score de urgência composto
    urgency = (
        severity * 0.35
        + security_kw * 0.20
        + system_crit * 0.15
        + sla_risk * 0.15
        + recurrence * 0.10
        + infra_kw * 0.05
    )

    # Fator de horário: fora do horário comercial aumenta urgência
    off_hours = ((hour < 0.25) | (hour > 0.75)).astype(float)
    urgency += off_hours * 0.10

    # Adicionar ruído realista
    noise = rng.normal(0, 0.05, n)
    urgency = np.clip(urgency + noise, 0, 1)

    # Classificação baseada em limiares
    labels = np.where(urgency >= 0.60, 2, labels)          # escalar
    labels = np.where((urgency >= 0.30) & (urgency < 0.60), 1, labels)  # responder
    labels = np.where(urgency < 0.30, 0, labels)           # ignorar

    return labels


def _balance_to_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    target_dist: list,
    n_total: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample to match the target class distribution.

    Reamostra os dados para corresponder à distribuição de classes alvo.
    Garante que o dataset tenha exatamente n_total amostras.
    """
    target_counts = [int(p * n_total) for p in target_dist]
    target_counts[-1] = n_total - sum(target_counts[:-1])

    new_features, new_labels = [], []
    for cls, count in enumerate(target_counts):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            idx = np.arange(len(labels))
        chosen = rng.choice(idx, size=count, replace=len(idx) < count)
        new_features.append(features[chosen])
        new_labels.append(np.full(count, cls))

    X = np.vstack(new_features)
    y = np.concatenate(new_labels)

    # Embaralhar
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def generate_dataset(
    cfg: DatasetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate train/test splits of the synthetic incident dataset.

    Gera as divisões de treino e teste do dataset sintético de incidentes.
    Retorna (X_train, X_test, y_train, y_test) como arrays numpy.
    """
    rng = np.random.default_rng(cfg.random_seed)

    # Gerar amostras brutas (extra para permitir balanceamento)
    n_raw = int(cfg.n_samples * 2.5)
    features = _generate_features(n_raw, rng)
    labels = _assign_labels(features, rng)

    # Balancear conforme distribuição alvo
    X, y = _balance_to_distribution(
        features, labels, cfg.class_distribution, cfg.n_samples, rng
    )

    # Dividir treino/teste
    split = int(cfg.n_samples * cfg.train_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test


def get_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """
    Compute class distribution statistics.

    Calcula estatísticas de distribuição das classes no dataset.
    Retorna um dicionário com contagens e percentuais por classe.
    """
    total = len(y)
    classes = [0, 1, 2]
    labels_pt = ["Ignorar", "Responder", "Escalar"]
    dist = {}
    for cls, label in zip(classes, labels_pt):
        count = int(np.sum(y == cls))
        dist[label] = {"count": count, "percent": round(count / total * 100, 1)}
    return dist


def get_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Convert feature matrix and labels to a pandas DataFrame.

    Converte a matriz de features e rótulos para um DataFrame pandas.
    Útil para análise exploratória e geração de relatórios.
    """
    df = pd.DataFrame(X, columns=FEATURE_NAMES_EN)
    df["label"] = y
    df["label_pt"] = df["label"].map({
        0: "Ignorar",
        1: "Responder",
        2: "Escalar",
    })
    return df
