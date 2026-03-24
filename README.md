# Assistente Médico com LLM Customizada

**Tech Challenge IADT — Fase 3** · Fine-tuning LoRA · LangChain LCEL · RAG + SQL · Guardrails + Auditoria

| | |
|---|---|
| **Demonstração** | [youtube.com/watch?v=vYB2mnsHh8c](https://youtu.be/vYB2mnsHh8c) |
| **Colab** | [Abrir notebook →](https://colab.research.google.com/drive/1tYl1FrsS4Z60Lhsbge5t9sP79scWwFcX?usp=sharing) · GPU T4 gratuita · ~20 min |
| **Relatório técnico** | [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) |

---

## Destaques

- **Fine-tuning com LoRA** — adapta um modelo 7B em ~20 min numa GPU T4 gratuita, treinando apenas ~1% dos parâmetros
- **Contexto híbrido (RAG + SQL)** — combina protocolos institucionais com dados reais do paciente na mesma resposta
- **Guardrails antes do LLM** — bloqueia solicitações impróprias *antes* de processar, não depois
- **Auditoria estruturada** — cada inferência gera um log JSON rastreável para compliance
- **Pipeline LangChain LCEL** — composável, testável e fácil de estender

---

## O que este projeto faz

O assistente recebe uma pergunta clínica e um `patient_id`, busca simultaneamente os **protocolos institucionais relevantes** (RAG) e o **contexto real do paciente** (SQL — alergias, exames pendentes, histórico), e gera uma resposta contextualizada com fontes citadas e validação humana obrigatória.

```
Pergunta clínica + patient_id
        │
        ▼
┌─────────────────────┐
│   Guardrails (in)   │  ← bloqueia antes de processar
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 SQL Query     RAG Search
(prontuário)  (protocolos)
    │            │
    └─────┬──────┘
          ▼
  ┌───────────────┐
  │  LLM + LoRA   │  ← fine-tuned em dados internos
  └───────┬───────┘
          ▼
┌─────────────────────┐
│  Guardrails (out)   │  ← valida resposta gerada
└─────────┬───────────┘
          ▼
  Resposta + fontes
  + alerta de validação
  + audit log (JSON)
```

---

## Quickstart

**Pré-requisitos:** Python 3.11+ · GPU opcional (só para fine-tuning)

```bash
pip install -r requirements.txt
cp .env.example .env

# Pipeline completo: dados → avaliação (sem retreinar)
python -m src.run_academic_pipeline

# Incluir fine-tuning LoRA
python -m src.run_academic_pipeline --run-train

# Só o assistente
python -m src.main
```

> Quer reproduzir sem configurar ambiente local? Use o **[notebook no Colab](https://colab.research.google.com/drive/1tYl1FrsS4Z60Lhsbge5t9sP79scWwFcX?usp=sharing)** — setup, treino e avaliação em 3 células.

---

## Resultados

Avaliados em conjunto acadêmico + blind set + protocol blind set:

| Métrica | Resultado |
|---|---|
| Guardrail accuracy | ~99%+ |
| Source coverage | 100% |
| Response rate | 100% |

Relatórios completos em `artifacts/` — `eval_report.json`, `blind_eval_report.json`, `protocol_blind_eval_report.json`.

---

## Stack

| Componente | Tecnologia | Localização |
|---|---|---|
| Fine-tuning | PEFT LoRA | `src/finetune/train_lora.py` |
| Orquestração | LangChain LCEL | `src/assistant/medical_assistant.py` |
| Recuperação | BM25 + Similarity Search | `src/assistant/knowledge_base.py` |
| Contexto do paciente | SQLAlchemy + SQLite | `src/assistant/patient_repository.py` |
| Segurança | Regex + Keywords | `src/security/guardrails.py` |
| Observabilidade | JSON Lines | `src/observability/audit_logger.py` |
| Avaliação | Métricas acadêmicas | `src/evaluation/` |

---

## Por que essas decisões?

**LoRA em vez de full fine-tuning**
Treina apenas ~1% dos parâmetros de um modelo 7B. Ciclo completo em GPU T4 do Colab em ~20 minutos. Qualidade de 85–90% do fine-tuning completo, com infraestrutura acessível. Pragmatismo > perfeição.

**RAG + SQL combinados**
RAG sozinho não conhece o paciente. SQL sozinho não conhece os protocolos. Juntos, a resposta é clinicamente contextualizada — alergias, exames pendentes e guidelines institucionais no mesmo prompt.

**Guardrails antes do LLM**
Bloquear uma solicitação imprópria *antes* de processar é mais barato, mais seguro e mais rápido do que filtrar a saída depois. Validar primeiro, gerar depois.

**Logs em JSON Lines**
Permite busca estruturada, análise de padrões e auditoria de compliance sem quebrar queries ao adicionar novos campos. Logging em texto livre não oferece nenhuma dessas garantias.

---

## Limites e responsabilidades

- O assistente **não substitui decisão médica humana**. Toda resposta inclui alerta de validação clínica obrigatória.
- Os dados neste repositório são **sintéticos**, gerados para fins acadêmicos.
- Para uso em produção: anonimização consistente + validação jurídica/LGPD + governança clínica ativa.

---

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | Relatório técnico: fine-tuning, arquitetura, avaliação |
| [`docs/COLAB_NOTEBOOK.md`](docs/COLAB_NOTEBOOK.md) | Guia de reprodução no Colab |
| [`docs/arquitetura_assistente_medico.html`](docs/arquitetura_assistente_medico.html) | Diagrama do fluxo LangChain renderizado |
| [`docs/INDEX.md`](docs/INDEX.md) | Índice geral de entregáveis |
