"""
main.py — Pipeline principal do Agente Inteligente de Triagem de Incidentes.
"""

import sys
import os
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sem display — salva PNG sem plt.show()
import matplotlib.pyplot as plt

# Adicionar o diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from src.data.dataset import generate_dataset, get_class_distribution, get_dataframe
from src.agent.triage_agent import TriageAgent

# ─────────────────────────────────────────
# CONSTANTES DE APRESENTAÇÃO
# ─────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     AGENTE INTELIGENTE DE TRIAGEM DE INCIDENTES DE TI            ║
║                                                                  ║
║  Paradigma Conexionista (RNA) + Paradigma Evolucionário (AG)     ║
║  Classificação automática de tickets em 3 categorias             ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Tickets de exemplo para demonstração (8 features normalizadas)
EXAMPLE_TICKETS = [
    {
        "descricao": "Servidor de produção fora do ar às 3h da madrugada (possível ataque)",
        "features": np.array([0.95, 0.15, 1.0, 0.9, 0.8, 0.7, 0.8, 0.95]),
    },
    {
        "descricao": "Impressora do escritório não está imprimindo (dia útil)",
        "features": np.array([0.1, 0.5, 0.1, 0.0, 0.1, 0.1, 0.0, 0.05]),
    },
    {
        "descricao": "Lentidão no sistema de homologação durante horário comercial",
        "features": np.array([0.4, 0.55, 0.5, 0.0, 0.5, 0.3, 0.4, 0.4]),
    },
    {
        "descricao": "Alerta de intrusão no firewall — possível ransomware",
        "features": np.array([0.85, 0.2, 0.9, 1.0, 0.9, 0.5, 0.6, 0.9]),
    },
    {
        "descricao": "Solicitação de reset de senha de usuário comum",
        "features": np.array([0.05, 0.6, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0]),
    },
]


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


def format_confusion_matrix(cm: np.ndarray, labels: list) -> str:
    """Format confusion matrix as ASCII table."""
    lines = []
    header = "          " + "  ".join(f"{l:>8}" for l in labels)
    lines.append(header)
    lines.append("          " + "-" * (len(labels) * 10))
    for i, row_label in enumerate(labels):
        row = f"{row_label:>8} |" + "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def save_evolution_chart(history: dict, best_generation: int, config, path: str) -> None:
    """Save fitness evolution chart to PNG file."""
    generations = range(1, len(history["best_fitness"]) + 1)
    best = np.array(history["best_fitness"])
    mean = np.array(history["mean_fitness"])
    std = np.array(history["std_fitness"])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Área sombreada: média ± desvio padrão
    ax.fill_between(
        generations,
        np.clip(mean - std, 0, 1),
        np.clip(mean + std, 0, 1),
        alpha=0.25,
        color="#a8dadc",
        label="Intervalo ±1σ",
    )

    # Linha do fitness médio
    ax.plot(
        generations,
        mean,
        "--",
        color="#a8dadc",
        linewidth=1.5,
        alpha=0.8,
        label="Fitness Médio",
    )

    # Linha do melhor fitness
    ax.plot(
        generations,
        best,
        "-",
        color="#4cc9f0",
        linewidth=2.5,
        label="Melhor Fitness",
    )

    # Marcador da geração do melhor resultado
    best_val = max(best)
    ax.axvline(
        x=best_generation,
        color="#f72585",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Melhor Geração ({best_generation})",
    )
    ax.scatter(
        [best_generation],
        [best_val],
        color="#f72585",
        s=100,
        zorder=5,
    )

    # Configurações estéticas
    ax.set_title(
        "Evolução do Fitness ao Longo das Gerações\nAlgoritmo Genético — Otimização de Pesos da RNA",
        color="white",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Geração", color="white", fontsize=12)
    ax.set_ylabel("Fitness (F1-score Ponderado)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#4cc9f0")
    ax.spines["left"].set_color("#4cc9f0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, len(generations))
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_text_report(
    config,
    dist_train: dict,
    dist_test: dict,
    metrics: dict,
    history: dict,
    best_generation: int,
    best_fitness: float,
    example_results: list,
    elapsed: float,
    path: str,
) -> None:
    """Save complete text report in Brazilian Portuguese."""
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cm = metrics["confusion_matrix"]
    class_labels = config.dataset.class_names_report

    lines = [
        "=" * 70,
        "  RELATÓRIO FINAL — AGENTE INTELIGENTE DE TRIAGEM DE INCIDENTES",
        f"  Data e Hora: {now}",
        "=" * 70,
        "",
        "1. RESUMO DO DATASET",
        "-" * 40,
        f"   Total de amostras: {config.dataset.n_samples}",
        f"   Divisão treino/teste: {int(config.dataset.train_split * 100)}% / "
        f"{int((1 - config.dataset.train_split) * 100)}%",
        "",
        "   Distribuição no conjunto de TREINO:",
    ]
    for label, info in dist_train.items():
        lines.append(f"     {label}: {info['count']} amostras ({info['percent']}%)")

    lines += [
        "",
        "   Distribuição no conjunto de TESTE:",
    ]
    for label, info in dist_test.items():
        lines.append(f"     {label}: {info['count']} amostras ({info['percent']}%)")

    lines += [
        "",
        "2. ARQUITETURA DA RNA",
        "-" * 40,
        f"   Camada de Entrada: {config.rna.input_size} neurônios (features)",
        f"   Camada Oculta: {config.rna.hidden_size} neurônios (ativação: tanh)",
        f"   Camada de Saída: {config.rna.output_size} neurônios (ativação: softmax)",
        f"   Total de parâmetros: "
        f"{config.rna.input_size * config.rna.hidden_size + config.rna.hidden_size + config.rna.hidden_size * config.rna.output_size + config.rna.output_size}",
        "",
        "3. CONFIGURAÇÃO DO ALGORITMO GENÉTICO",
        "-" * 40,
        f"   Tamanho da população: {config.ag.population_size}",
        f"   Número de gerações: {config.ag.generations}",
        f"   Taxa de mutação: {config.ag.mutation_rate}",
        f"   Força de mutação: {config.ag.mutation_strength}",
        f"   Taxa de crossover: {config.ag.crossover_rate}",
        f"   Tamanho do torneio: {config.ag.tournament_size}",
        f"   Elitismo (top N): {config.ag.elitism_count}",
        f"   Penalidade crítica: {config.critical_penalty}",
        "",
        "4. RESULTADOS DA EVOLUÇÃO",
        "-" * 40,
        f"   Melhor fitness encontrado: {best_fitness:.4f}",
        f"   Geração do melhor resultado: {best_generation}/{config.ag.generations}",
        f"   Fitness inicial (geração 1): {history['best_fitness'][0]:.4f}",
        f"   Fitness final (última geração): {history['best_fitness'][-1]:.4f}",
        f"   Tempo total de execução: {elapsed:.1f} segundos",
        "",
        "5. MÉTRICAS DE AVALIAÇÃO NO CONJUNTO DE TESTE",
        "-" * 40,
        f"   Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)",
        f"   F1-Score Ponderado: {metrics['f1_weighted']:.4f}",
        "",
        "   Relatório de Classificação:",
        metrics["classification_report"],
        "",
        "6. MATRIZ DE CONFUSÃO",
        "-" * 40,
        "   (Linhas = real, Colunas = predito)",
        "",
        format_confusion_matrix(cm, class_labels),
        "",
        "7. EXEMPLOS DE PREDIÇÃO",
        "-" * 40,
    ]

    for i, result in enumerate(example_results[:3], 1):
        lines += [
            f"   Exemplo {i}: {result['descricao']}",
            f"     Features: {[f'{v:.2f}' for v in result['features']]}",
            f"     Decisão: {result['label_pt']} (confiança: {result['confidence']:.1f}%)",
            "",
        ]

    lines += [
        "=" * 70,
        "  FIM DO RELATÓRIO",
        "=" * 70,
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def main() -> None:
    """Main execution pipeline for the triage agent."""
    start_time = time.time()

    # ── 1. Banner ──────────────────────────────────────────────────
    print(BANNER)

    # ── 2. Configuração ───────────────────────────────────────────
    config = get_config()
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.reports_dir, exist_ok=True)

    # ── 3. Dataset ────────────────────────────────────────────────
    print_section("GERAÇÃO DO DATASET SINTÉTICO")
    print(f"  Gerando {config.dataset.n_samples} tickets de incidentes...")
    X_train, X_test, y_train, y_test = generate_dataset(config.dataset)

    dist_train = get_class_distribution(y_train)
    dist_test = get_class_distribution(y_test)

    print(f"\n  Conjunto de TREINO ({len(y_train)} amostras):")
    for label, info in dist_train.items():
        bar = "█" * int(info["percent"] / 3)
        print(f"    {label:<22} {info['count']:>4} ({info['percent']:>5.1f}%) {bar}")

    print(f"\n  Conjunto de TESTE ({len(y_test)} amostras):")
    for label, info in dist_test.items():
        bar = "█" * int(info["percent"] / 3)
        print(f"    {label:<22} {info['count']:>4} ({info['percent']:>5.1f}%) {bar}")

    # ── 4. Treinamento ────────────────────────────────────────────
    print_section("TREINAMENTO — OTIMIZAÇÃO EVOLUTIVA DOS PESOS DA RNA")
    print(f"  Configuração do AG:")
    print(f"    • População: {config.ag.population_size} indivíduos")
    print(f"    • Gerações: {config.ag.generations}")
    print(f"    • Taxa de mutação: {config.ag.mutation_rate}")
    print(f"    • Elitismo: top {config.ag.elitism_count}\n")
    print(f"  Iniciando evolução...\n")

    agent = TriageAgent(config)
    history = agent.train(X_train, y_train, X_test, y_test)

    print(f"\n{agent.get_evolution_summary()}")

    # ── 5. Avaliação ──────────────────────────────────────────────
    print_section("AVALIAÇÃO NO CONJUNTO DE TESTE")
    metrics = agent.evaluate(X_test, y_test)

    print(f"  Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"  F1-Score Ponderado: {metrics['f1_weighted']:.4f}")
    print(f"\n  Relatório de Classificação:")
    print("  " + metrics["classification_report"].replace("\n", "\n  "))

    print(f"  Matriz de Confusão (linhas=real, colunas=predito):")
    cm = metrics["confusion_matrix"]
    labels = config.dataset.class_names_report
    print("  " + format_confusion_matrix(cm, labels).replace("\n", "\n  "))

    # ── 6. Exemplos de predição ───────────────────────────────────
    print_section("EXEMPLOS DE PREDIÇÃO — TICKETS REAIS")
    example_results = []
    for ticket in EXAMPLE_TICKETS:
        label_int, label_pt, confidence = agent.predict(ticket["features"])
        result = {
            **ticket,
            "label_int": label_int,
            "label_pt": label_pt,
            "confidence": confidence,
        }
        example_results.append(result)

        icon = "🔴" if label_int == 2 else ("🟡" if label_int == 1 else "🟢")
        print(f"\n  {icon} {ticket['descricao']}")
        print(f"     Features: {[f'{v:.2f}' for v in ticket['features']]}")
        print(f"     ➜ Decisão: [{label_pt}] — Confiança: {confidence:.1f}%")

    # ── 7. Gráfico de evolução ────────────────────────────────────
    print_section("GERANDO GRÁFICO E RELATÓRIO")
    print(f"  Salvando gráfico de evolução...")
    save_evolution_chart(
        history,
        agent.best_generation,
        config,
        config.chart_path,
    )
    print(f"  ✓ Gráfico salvo em: {config.chart_path}")

    # ── 8. Salvar modelo ──────────────────────────────────────────
    agent.save_model(config.model_path)
    print(f"  ✓ Modelo salvo em: {config.model_path}")

    # ── 9. Relatório final ────────────────────────────────────────
    elapsed = time.time() - start_time
    best_fitness = max(history["best_fitness"])

    save_text_report(
        config=config,
        dist_train=dist_train,
        dist_test=dist_test,
        metrics=metrics,
        history=history,
        best_generation=agent.best_generation,
        best_fitness=best_fitness,
        example_results=example_results,
        elapsed=elapsed,
        path=config.report_path,
    )
    print(f"  ✓ Relatório salvo em: {config.report_path}")

    # ── 10. Tempo total ───────────────────────────────────────────
    print_section("EXECUÇÃO CONCLUÍDA")
    print(f"  ⏱  Tempo total de execução: {elapsed:.1f} segundos")
    print(f"  🏆 Melhor fitness atingido: {best_fitness:.4f}")
    print(f"  🎯 Acurácia no teste: {metrics['accuracy'] * 100:.1f}%")
    print(f"\n  Arquivos gerados:")
    print(f"    • {config.chart_path}")
    print(f"    • {config.model_path}")
    print(f"    • {config.report_path}")
    print()


if __name__ == "__main__":
    main()
