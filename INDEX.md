# Documentação Técnica
## Tech Challenge IADT - Fase 3

---

## Entregáveis Conforme Requisitos

### 1. TECHNICAL_REPORT.md
Relatório técnico detalhado com:
- **Explicação do processo de fine-tuning** (Seção 4)
- **Descrição do assistente médico criado** (Seção 5)
- **Diagrama do fluxo LangChain** (Seção 5.2)
- **Avaliação do modelo e análise dos resultados** (Seção 7)

### 2. ARCHITECTURE_DIAGRAM.md
Diagrama visual do fluxo LangChain

### 3. Código-fonte (Repositório Git)
- **Pipeline de fine-tuning:** `src/finetune/train_lora.py`
- **Integração com LangChain:** `src/assistant/medical_assistant.py` (LCEL chains)
- **Fluxos do LangGraph:** `src/assistant/workflow.py`
- **Dataset anonimizado:** `data/processed/` (protocolos + MedQuAD)

---

## Estrutura do Código

```
src/
├── finetune/
│   └── train_lora.py ............... Pipeline de fine-tuning LoRA
├── assistant/
│   ├── medical_assistant.py ........ Integração LangChain (LCEL chains)
│   ├── workflow.py ................. Fluxos de orquestração
│   ├── knowledge_base.py ........... RAG search
│   └── patient_repository.py ....... SQL context
├── data/
│   ├── convert_medquad.py .......... Ingestão de dados
│   └── build_training_dataset.py ... Dataset preparation
├── security/
│   └── guardrails.py ............... Validações
└── evaluation/
    └── evaluate_assistant.py ....... Avaliação de resultados

data/
├── raw/ ............................ Dados brutos (MedQuAD)
├── processed/ ...................... Datasets de treinamento
└── eval/ ........................... Conjuntos de avaliação

models/
└── medical_assistant_lora/ ......... Modelo fine-tuned (LoRA adapter)
```

---

## Como Executar

```bash
# Setup
pip install -r requirements.txt

# Pipeline completo
python -m src.run_academic_pipeline

# Com MedQuAD do Hugging Face
python -m src.run_academic_pipeline --medquad-source hf

# Com fine-tuning (recomendado: Colab)
python -m src.run_academic_pipeline --run-train
```

---

## Arquivos de Referência

- **TECHNICAL_REPORT.md** - Documentação técnica completa com metodologia
- **DEVELOPMENT_NOTES.md** - Notas de desenvolvimento
- **VIDEO_SCRIPT.md** - Script completo para vídeo demonstrativo (15 min)
- **README.md** - Instruções de setup
