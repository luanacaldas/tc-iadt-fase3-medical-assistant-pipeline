# 🚀 Executar Pipeline no Google Colab

## Opção 1: Clique direto (Recomendado)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luanacaldas/tc-iadt-fase3-medical-assistant-pipeline/blob/main/colab_pipeline.ipynb)

## Opção 2: Copie no Colab

Cole este código em um novo Colab Notebook, célula por célula:

### Célula 1: Clonar repositório
```python
!git clone https://github.com/luanacaldas/tc-iadt-fase3-medical-assistant-pipeline.git
%cd tc-iadt-fase3-medical-assistant-pipeline
```

### Célula 2: Corrigir versão de requests (conflito Colab)
```python
!pip install requests==2.32.4 --quiet
```

### Célula 3: Instalar dependências
```python
!pip install -r requirements.txt --quiet
```

### Célula 4: Verificar ambiente
```python
import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Célula 5: Rodar pipeline completo (30-60 min com GPU T4)
```python
!python -m src.run_academic_pipeline --run-train
```

### Célula 6: Baixar resultados
```python
from google.colab import files
import os

# Baixar relatórios de avaliação
if os.path.exists('artifacts/protocol_blind_eval_report.json'):
    files.download('artifacts/protocol_blind_eval_report.json')
    files.download('artifacts/blind_eval_report.json')
    files.download('artifacts/eval_report.json')

# Baixar modelo treinado (LoRA)
if os.path.exists('models/medical_assistant_lora'):
    !zip -r models.zip models/medical_assistant_lora/
    files.download('models.zip')
    
print("✅ Download concluído!")
```

### Célula 7 (Opcional): Testar assistente
```python
from src.assistant.knowledge_base import MedicalAssistant

assistant = MedicalAssistant(use_lora=True)

# Teste interativo
pergunta = input("Digite sua pergunta: ")
resposta = assistant.query(pergunta)
print(f"\n✅ Resposta:\n{resposta}")
```

---

## ⚙️ Configuração Colab

**Importante**: Mude para GPU antes de rodar!

1. Clique em `Configurações do runtime` (ou ⚙️)
2. Selecione `T4 GPU` em "Acelerador de hardware"
3. Clique `Salvar`
4. Execute as células na ordem

---

## 📊 Saída esperada

Após os ~45 min de execução:
- ✅ `artifacts/protocol_blind_eval_report.json` - Métricas finais
- ✅ `artifacts/blind_eval_report.json` - Avaliação cega
- ✅ `artifacts/eval_report.json` - Avaliação de guardrails
- ✅ `models/medical_assistant_lora/` - Modelo fine-tuned com LoRA

---

## 🏆 Resultado: "100% Produção"

✅ **Python 3.12** + **GPU T4 (Colab)** validado  
✅ **Fine-tuning LoRA** completo  
✅ **Métricas acadêmicas** comprovadas  
✅ **Modelo deployável** em produção  

Próximo passo (opcional): substituir dados fictícios em `data/raw/internal_clinical_qa.jsonl` por **dados reais do hospital** (anonimizados).
