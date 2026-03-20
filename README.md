# Tech Challenge IADT - Fase 3

Projeto base para o desafio de **assistente médico com LLM customizada + LangChain**, cobrindo:

1. Fine-tuning com dados internos (pipeline pronto para LoRA).
2. Assistente clínico com LangChain (RAG + SQL + contexto do paciente).
3. Segurança, validação humana, logging e explainability.

## Decisão prática: Colab ou VS Code?

### Recomendação objetiva
- **Continue no VS Code** para desenvolver o sistema completo (arquitetura, LangChain, segurança, logs, banco).
- Use **Colab apenas para o treinamento pesado** (fine-tuning) se faltar GPU local.

Isso facilita porque o desafio não é só treinar modelo: também exige integração, auditoria e fluxo clínico automatizado.

---

## Arquitetura proposta

- `src/data/`: ingestão, curadoria, anonimização e geração de dataset de treino.
- `src/finetune/`: script de fine-tuning LoRA (Hugging Face/PEFT).
- `src/assistant/`: orquestração LangChain (RAG + SQL + guardrails).
- `src/security/`: regras de segurança e limites de atuação.
- `src/observability/`: logging estruturado e trilha de auditoria.
- `src/main.py`: execução de exemplo ponta-a-ponta.

Fluxo principal:
1. Recebe dados do paciente.
2. Consulta pendências de exame no banco SQL.
3. Recupera protocolos internos relevantes via RAG.
4. Gera sugestão clínica com LLM.
5. Aplica validações de segurança.
6. Retorna resposta com fontes + alerta de validação humana.
7. Registra tudo em log de auditoria.

---

## Requisitos

- Python 3.11+
- (Opcional) GPU para fine-tuning

## Setup

1. Criar ambiente virtual e instalar dependências:

```bash
pip install -r requirements.txt
```

2. Copiar variáveis de ambiente:

```bash
copy .env.example .env
```

3. Executar demonstração local:

```bash
python -m src.main
```

## Pipeline sugerido de execução

### Opção 1: tudo em um comando (recomendado para entrega)

```bash
# Usa MedQuAD local (XML), gera split, roda avaliações e pula treino pesado por padrão
python -m src.run_academic_pipeline

# Para usar MedQuAD do Hugging Face
python -m src.run_academic_pipeline --medquad-source hf

# Para incluir fine-tuning LoRA no pipeline
python -m src.run_academic_pipeline --run-train
```

1. Converter MedQuAD para JSONL (escolha 1 opção):

```bash
# Opção A (XML local já baixado)
python -m src.data.convert_medquad

# Opção B (Hugging Face - lavita/MedQuAD)
python -m src.data.convert_medquad --source hf
```

2. Gerar dataset combinado e anonimizado (interno + MedQuAD):

```bash
python -m src.data.build_training_dataset

# Exemplo para reforçar dados internos do hospital (recomendado para trabalho acadêmico)
python -m src.data.build_training_dataset --internal-multiplier 8 --medquad-multiplier 1

# Split treino/validação cego (sem vazamento por pergunta)
python -m src.data.build_training_dataset --validation-ratio 0.1 --internal-multiplier 8 --medquad-multiplier 1
```

3. Rodar fine-tuning LoRA (local com GPU ou Colab):

```bash
python -m src.finetune.train_lora
```

4. Rodar assistente com workflow clínico e auditoria:

```bash
python -m src.main
```

5. Rodar avaliação acadêmica (segurança + fontes + rastreabilidade):

```bash
python -m src.evaluation.build_eval_set
python -m src.evaluation.evaluate_assistant

# Avaliação cega com perguntas reais do dataset (estratificada por temas clínicos)
python -m src.evaluation.build_blind_eval_set --validation-ratio 0.02 --max-cases 80 --focused-per-source 10
python -m src.evaluation.evaluate_assistant --eval-set data/eval/blind_eval_set.jsonl --report artifacts/blind_eval_report.json

# Blind Protocol Set (30 casos inéditos focados nos protocolos internos)
python -m src.evaluation.build_protocol_blind_set
python -m src.evaluation.evaluate_assistant --eval-set data/eval/protocol_blind_eval_set.jsonl --report artifacts/protocol_blind_eval_report.json
```

O script cria um conjunto com 30 casos (sepse, dor torácica e segurança clínica),
permitindo medir de forma mais robusta cobertura de fontes e guardrails.

## Como incluir as ~160 perguntas internas do grupo

- Coloque os novos pares em `data/raw/internal_clinical_qa.jsonl` no formato:

```json
{"input":"pergunta", "output":"resposta", "source":"Protocolo X", "domain":"specialty"}
```

- O comando `python -m src.data.build_training_dataset` já combina automaticamente:
  - `data/raw/internal_clinical_qa.jsonl`
  - `data/raw/medquad_qa.jsonl`

## Estratégia acadêmica para dataset menor

- Mesmo com ~16.4k pares no MedQuAD, o ganho principal vem de adaptar ao contexto local.
- Use oversampling dos dados internos para aumentar aderência institucional no fine-tuning.
- Reporte métricas mínimas no trabalho:
  - `guardrail_accuracy` (bloqueio de pedidos impróprios);
  - `source_coverage` (respostas com fontes rastreáveis);
  - `non_empty_rate` (robustez básica de resposta).
- Meta prática para apresentação: `source_coverage >= 0.80` e `guardrail_accuracy >= 0.95`.

---

## Como isso atende os requisitos obrigatórios

### 1) Fine-tuning de LLM com dados médicos internos
- Pipeline de preprocessing/anonimização em `src/data/preprocess.py`.
- Conversão e curadoria do MedQuAD em `src/data/convert_medquad.py`.
- Curadoria e montagem de dataset instrucional em `src/data/build_training_dataset.py`.
- Script de fine-tuning LoRA em `src/finetune/train_lora.py`.

### 2) Assistente médico com LangChain
- Pipeline integrado em `src/assistant/medical_assistant.py`.
- Consulta SQL (prontuário/pendências) em `src/assistant/patient_repository.py`.
- Fluxo de decisão automatizado em `src/assistant/workflow.py`.
- Contextualização por paciente + documentos internos no prompt.

### 3) Segurança e validação
- Guardrails em `src/security/guardrails.py`.
- Logging e auditoria em `src/observability/audit_logger.py`.
- Explainability via citação explícita de fontes na resposta.

---

## Importante para apresentação

- Este projeto já vem com **dados sintéticos** para demonstração.
- Para produção, usar dados reais apenas após:
  - anonimização forte;
  - validação jurídica/LGPD;
  - governança clínica.
- O assistente **não prescreve automaticamente**: sempre exige validação humana.
