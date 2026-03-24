# Índice de Documentação

## Tech Challenge IADT - Fase 3

---

## Entregáveis Principais

### 1. TECHNICAL_REPORT.md

Relatório técnico detalhado com:

- **Explicação do processo de fine-tuning** (Seção 4)
- **Descrição do assistente médico criado** (Seção 5)
- **Diagrama do fluxo LangChain** (Seção 5.2)
- **Avaliação do modelo e análise dos resultados** (Seção 7)

### 2. arquitetura_assistente_medico.mmd e arquitetura_assistente_medico.html

Diagrama visual do fluxo LangChain (fonte Mermaid + versão renderizada)

### 3. Código-fonte

- **Pipeline de fine-tuning:** `src/finetune/train_lora.py`
- **Integração com LangChain:** `src/assistant/medical_assistant.py` (LCEL chains)
- **Fluxos de workflow clínico:** `src/assistant/workflow.py`
- **Dataset anonimizado:** `data/processed/` (protocolos + MedQuAD)

### 4. Reprodução e Demonstração

- **../README.md:** setup local e pipeline completo
- **COLAB_NOTEBOOK.md:** execução guiada no Colab
- **Vídeo de demonstração:** https://youtu.be/vYB2mnsHh8c
- **Google Colab:** https://colab.research.google.com/drive/1tYl1FrsS4Z60Lhsbge5t9sP79scWwFcX?usp=sharing

---

## Estrutura Atual do Projeto

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

artifacts/
└── *.json .......................... Relatórios de avaliação
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
- **COLAB_NOTEBOOK.md** - Guia de execução no Colab
- **arquitetura_assistente_medico.mmd** - Diagrama editável em Mermaid
- **arquitetura_assistente_medico.html** - Diagrama renderizado em HTML
- **Vídeo de demonstração:** https://youtu.be/vYB2mnsHh8c
- **Google Colab:** https://colab.research.google.com/drive/1tYl1FrsS4Z60Lhsbge5t9sP79scWwFcX?usp=sharing
- **../README.md** - Instruções de setup
