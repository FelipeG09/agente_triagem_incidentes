# Agente Inteligente de Triagem de Incidentes

## Descrição

Sistema de inteligência artificial para triagem automatizada de tickets de incidentes de TI. Combina dois paradigmas distintos de IA — um **Modelo Conexionista (Rede Neural Artificial)** e um **Modelo Evolucionário (Algoritmo Genético)** — para classificar automaticamente incidentes em três categorias de prioridade: **Ignorar**, **Responder** e **Escalar Imediatamente**.

## Objetivo do Projeto

Em ambientes de TI corporativos, a equipe de suporte recebe dezenas a centenas de tickets por dia. A triagem manual é lenta, sujeita a erros humanos e cria gargalos críticos — especialmente durante incidentes de segurança fora do horário comercial. Este agente automatiza essa decisão com base em 8 features objetivas do ticket, garantindo que incidentes críticos nunca sejam ignorados.

---

## Paradigmas de IA Utilizados

### Paradigma Conexionista — A Rede Neural Artificial (RNA)

A RNA é o **classificador central** do sistema. Implementada do zero com NumPy puro (sem frameworks como PyTorch ou TensorFlow), ela possui a seguinte arquitetura:

```
Entrada (8 neurônios)
    ↓  [pesos W1 + bias b1]
Camada Oculta (12 neurônios) — ativação: tanh
    ↓  [pesos W2 + bias b2]
Camada de Saída (3 neurônios) — ativação: softmax
    ↓
Probabilidades para [Ignorar | Responder | Escalar]
```

**Entradas (8 features normalizadas 0.0–1.0):**
| # | Feature | Descrição |
|---|---------|-----------|
| 1 | `severity` | Severidade declarada do ticket |
| 2 | `hour_normalized` | Horário (0=meia-noite, 1=23h) |
| 3 | `system_criticality` | Produção=1.0, Homolog=0.5, Dev=0.1 |
| 4 | `security_keyword` | Presença de termos de segurança |
| 5 | `infra_keyword` | Presença de termos de infraestrutura |
| 6 | `sender_frequency` | Volume de tickets do remetente hoje |
| 7 | `recurrence` | É um problema recorrente? |
| 8 | `response_time_sla` | Risco de violação de SLA |

**Saídas:**
- `0` → Ignorar (baixa prioridade)
- `1` → Responder (média prioridade)
- `2` → Escalar Imediatamente (alta prioridade)

**Por que tanh na camada oculta?**
A tangente hiperbólica é centrada em zero (retorna valores em (-1, 1)), o que facilita a convergência durante a otimização e permite que a rede represente tanto ativações positivas quanto negativas.

**Por que softmax na saída?**
A softmax transforma os logits brutos em uma distribuição de probabilidade (soma = 1), permitindo interpretar a saída como confiança por classe.

**Por que não usar backpropagation?**
Intencionalmente. Os pesos são otimizados exclusivamente pelo Algoritmo Genético, demonstrando que é possível treinar redes neurais sem gradientes — paradigma aplicável quando a função de perda não é diferenciável ou quando se deseja explorar o espaço de soluções de forma global.

---

### Paradigma Evolucionário — O Algoritmo Genético (AG)

O AG é o **motor de aprendizado** que substitui o backpropagation. Ele evolui uma população de vetores de pesos candidatos ao longo de múltiplas gerações.

**Componentes do AG:**

**1. Representação (Cromossomo)**
Cada indivíduo da população é um vetor NumPy 1D contendo todos os pesos e biases da RNA concatenados (W1 + b1 + W2 + b2). Para a arquitetura 8→12→3, isso resulta em **135 parâmetros** por indivíduo.

**2. Inicialização da População**
100 indivíduos com pesos inicializados por distribuição normal N(0, 0.5), garantindo diversidade inicial.

**3. Função de Fitness**
Avalia a qualidade de cada conjunto de pesos (veja seção dedicada abaixo).

**4. Seleção por Torneio**
Para cada novo indivíduo filho, sorteiam-se 5 indivíduos da população e o de maior fitness é escolhido como pai. Vantagens sobre a roleta: independente da escala absoluta do fitness, pressão seletiva controlável, eficiente computacionalmente.

**5. Cruzamento Uniforme (Crossover)**
Cada gene (peso) do filho é herdado independentemente de um dos dois pais com probabilidade 0.75. Preserva mais diversidade genética que cruzamentos de ponto único.

**6. Mutação Gaussiana**
Cada gene é mutado com probabilidade 0.08, recebendo ruído N(0, 0.2). Permite exploração local do espaço de soluções.

**7. Elitismo**
Os 5 melhores indivíduos de cada geração passam intactos para a próxima. Garante que o melhor fitness nunca decresce ao longo das gerações.

---

### Integração Híbrida

Este é o conceito central do projeto. A RNA e o AG trabalham juntos da seguinte forma:

```
AG gerencia uma população de vetores de pesos
         ↓
Para cada indivíduo (vetor de pesos):
  → Injeta os pesos na RNA via set_weights()
  → Passa o dataset de treino pelo forward pass
  → Computa o fitness com base nas predições
         ↓
Seleciona os melhores → Crossover → Mutação
         ↓
Nova geração → Repete por 150 gerações
         ↓
Melhores pesos encontrados → Configura a RNA final
         ↓
Agente pronto para predição em produção
```

O AG fornece **otimização global** (sem risco de mínimos locais do gradiente descendente), enquanto a RNA fornece **capacidade de generalização** através de sua arquitetura não-linear.

---

## Arquitetura do Projeto

```
agente-triagem-incidentes/
├── config.py                   # Todos os hiperparâmetros centralizados (dataclasses)
├── main.py                     # Pipeline principal: dataset → treino → avaliação → relatório
├── requirements.txt            # Dependências (numpy, pandas, matplotlib, scikit-learn)
├── src/
│   ├── data/
│   │   └── dataset.py          # Gerador de dataset sintético com correlações realistas
│   ├── neural_network/
│   │   ├── activations.py      # relu, tanh, softmax implementados com numpy puro
│   │   └── network.py          # Classe NeuralNetwork (forward pass, get/set weights)
│   ├── genetic/
│   │   ├── algorithm.py        # Classe GeneticAlgorithm (loop evolutivo completo)
│   │   ├── fitness.py          # Função de fitness com penalidade crítica
│   │   ├── operators.py        # Crossover uniforme e mutação gaussiana
│   │   └── selection.py        # Seleção por torneio
│   └── agent/
│       └── triage_agent.py     # Classe TriageAgent (API de alto nível para uso externo)
├── results/
│   ├── evolucao_fitness.png    # Gráfico gerado automaticamente (gerado em execução)
│   └── best_model.pkl          # Modelo serializado (gerado em execução)
└── reports/
    └── relatorio_final.txt     # Relatório completo em PT-BR (gerado em execução)
```

---

## Instalação e Execução

**Pré-requisitos:** Python 3.9+

```bash
# 1. Clonar ou baixar o projeto
cd agente-triagem-incidentes

# 2. (Opcional) Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar o pipeline completo
python main.py
```

**Saída esperada:**
- Console: progresso da evolução geração a geração
- `results/evolucao_fitness.png`: gráfico da curva de evolução
- `results/best_model.pkl`: modelo treinado serializável
- `reports/relatorio_final.txt`: relatório completo

---

## Como Funciona — Fluxo Completo

1. **Carregamento da configuração** (`config.py`): todos os hiperparâmetros são lidos de dataclasses, sem números mágicos no código.

2. **Geração do dataset** (`src/data/dataset.py`): 1000 tickets sintéticos são criados com 8 features normalizadas. Correlações realistas são aplicadas — ex: alta severidade + keyword de segurança + horário noturno → provável "Escalar". O dataset é balanceado para a distribuição alvo (65% ignorar, 25% responder, 10% escalar).

3. **Divisão treino/teste**: 800 amostras para treino, 200 para avaliação final.

4. **Inicialização do TriageAgent**: cria instâncias da RNA (8→12→3) e do AG (100 indivíduos).

5. **Evolução (150 gerações)**:
   - Avalia todos os 100 indivíduos na base de treino
   - Registra best/mean/std fitness por geração
   - Preserva os 5 melhores (elitismo)
   - Seleciona pais por torneio (grupos de 5)
   - Gera filhos por crossover uniforme (taxa 0.75)
   - Aplica mutação gaussiana (taxa 0.08, força 0.2)
   - Repete

6. **Carregamento dos melhores pesos**: os pesos do melhor indivíduo encontrado em qualquer geração são injetados na RNA.

7. **Avaliação no conjunto de teste**: F1-score ponderado, acurácia, matriz de confusão, relatório por classe.

8. **Predições de exemplo**: 5 tickets pré-definidos são classificados com score de confiança.

9. **Geração de artefatos**: gráfico PNG da curva de evolução e relatório TXT completo.

---

## Função de Fitness

A função de fitness avalia a qualidade de um conjunto de pesos candidatos:

```
fitness(pesos) = F1_ponderado(predições, rótulos_reais)
              - 0.15 × (número de incidentes críticos classificados como "ignorar")
```

**Por que F1-score ponderado como base?**
O F1-score pondera a média harmônica de precisão e recall de cada classe pelo seu suporte (número de amostras). É mais robusto que a acurácia em datasets desbalanceados — evita que o modelo "aprenda" a classificar tudo como "Ignorar" (classe majoritária com 65%) e pareça ter boa performance.

**Por que a penalidade crítica?**
Em operações reais de TI, classificar um incidente crítico (ex: ataque ransomware, servidor de produção fora) como "ignorar" pode causar:
- Perda de dados corporativos irreversível
- Violação de SLA com multas contratuais
- Dano reputacional
- Comprometimento de segurança

A penalidade de -0.15 por incidente crítico ignorado força o AG a evoluir soluções que, mesmo que errem em outras classes, **nunca ignorem o que deveria ser escalado**.

---

## Resultados Esperados

Após 150 gerações de evolução:
- **Melhor fitness** > 0.75
- **Acurácia no teste** > 70%
- **Redução significativa** de falsos negativos críticos (label 2 predito como 0)
- **Curva de evolução** mostrando progresso contínuo do fitness

---

## Equivalentes no Mercado Real

Este projeto demonstra o mesmo conceito aplicado em soluções enterprise:

| Solução | Empresa | Aplicação |
|---------|---------|-----------|
| **ServiceNow ITSM AI** | ServiceNow | Triagem automática de tickets com ML, priorização por SLA |
| **Microsoft Sentinel** | Microsoft | SOAR com IA para classificação de alertas de segurança |
| **Splunk SOAR** | Splunk | Automação de resposta a incidentes com playbooks inteligentes |
| **PagerDuty AIOps** | PagerDuty | Roteamento inteligente de alertas e supressão de ruído |

A diferença deste projeto é a **transparência pedagógica**: ao invés de uma caixa-preta, cada componente é implementado do zero, tornando o sistema auditável e educativo.

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|-----------|--------|-----|
| Python | 3.9+ | Linguagem principal |
| NumPy | ≥1.24 | Álgebra linear, RNA, AG |
| Pandas | ≥2.0 | Análise exploratória do dataset |
| Matplotlib | ≥3.7 | Geração do gráfico de evolução |
| scikit-learn | ≥1.3 | Métricas de avaliação (F1, acurácia, relatório) |

**Sem PyTorch, TensorFlow, Keras ou qualquer framework de ML.**

---

## Estrutura de Classes

```
ProjectConfig (dataclass)
├── RNAConfig           — hiperparâmetros da rede neural
├── AGConfig            — hiperparâmetros do algoritmo genético
└── DatasetConfig       — configuração do dataset

NeuralNetwork
├── get_weights()       → np.ndarray (flat)
├── set_weights(flat)   → None
├── forward(X)          → probabilidades (n_samples, 3)
└── predict(X)          → rótulos (n_samples,)

GeneticAlgorithm
├── _initialize_population()  → List[np.ndarray]
├── _evaluate_population()    → List[float]
├── _apply_elitism()          → List[np.ndarray]
└── run(X_train, y_train, ...) → (best_weights, history)

TriageAgent
├── train(X_train, y_train, X_val, y_val) → history
├── evaluate(X_test, y_test)              → metrics_dict
├── predict(ticket_features)              → (int, str, float)
├── save_model(path)                      → None
└── load_model(path)                      → None
```

---

## Referências

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Mitchell, M. (1996). *An Introduction to Genetic Algorithms*. MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Yao, X. (1999). Evolving artificial neural networks. *Proceedings of the IEEE*, 87(9), 1423–1447.
- ServiceNow. (2024). *AI-Powered Incident Management*. https://www.servicenow.com
- Microsoft. (2024). *Microsoft Sentinel Documentation*. https://learn.microsoft.com/azure/sentinel
