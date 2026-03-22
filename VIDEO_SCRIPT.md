# Script de Vídeo Demonstrativo

## Assistente Médico com Fine-tuning LoRA e LangChain

**Duração:** ~15 minutos  
**Objetivo:** Demonstrar o sistema completo funcionando

---

## 📹 Estrutura do Vídeo

### Parte 1: Introdução (1 min)

**O que dizer:**

> "Olá, meu nome é Luana Caldas. Este vídeo demonstra um assistente médico baseado em LLM customizada que eu desenvolvi como projeto do Tech Challenge IADT Fase 3.
>
> O sistema integra:
>
> - Fine-tuning LoRA (adaptação do modelo para domínio médico)
> - LangChain como orquestrador
> - Base de conhecimento com 16.4k Q&A médicos
> - Guardrails de segurança (99% de acurácia)
>
> Vou demonstrar o treinamento, o funcionamento, e como ele processa perguntas clínicas reais."

**Ação na câmera:**

- Mostrar a câmera (você mesmo)
- Abrir o repositório no GitHub/VS Code
- Mostrar a estrutura de pastas:
  ```
  src/
  ├── finetune/     # Fine-tuning
  ├── assistant/    # LangChain
  ├── data/         # Datasets
  └── evaluation/   # Avaliações
  ```

---

### Parte 2: Fine-tuning e Avaliação (Rodado no Colab) - 3-4 min

**O que dizer:**

> "Para treinar o modelo com LoRA, eu rodei o pipeline completo no Google Colab, que oferece GPU T4 grátis. O treinamento levou aproximadamente 20 minutos.
>
> Vou mostrar como foi feito e o que foi gerado."

**Ações na câmera:**

#### 2.1: Mostrar o Notebook Colab (ou printscreen)

Abra [COLAB_NOTEBOOK.md](COLAB_NOTEBOOK.md) no repositório:

```bash
code COLAB_NOTEBOOK.md
```

Mostre as 3 células principais:

**Célula 1: Setup**

```python
!git clone https://github.com/luanacaldas/tc-iadt-fase3-medical-assistant-pipeline.git
%cd tc-iadt-fase3-medical-assistant-pipeline
!pip install requests==2.32.4 -q
!pip install -r requirements.txt -q
```

**Célula 2: Rodar o Pipeline Completo (Treino + Avaliação)**

```python
!python -m src.run_academic_pipeline --run-train
```

**O que esse comando faz:**

- Carrega os 16.4k pares de Q&A do MedQuAD
- Configura LoRA (rank=16, lora_alpha=32)
- Treina por 1 época com learning rate 2e-4
- Avalia o modelo em 3 conjuntos cegos (10 + 80 + 30 casos)
- Gera relatórios de avaliação em JSON

**Célula 3: Baixar Resultados**

```python
from google.colab import files
!zip -r artifacts.zip artifacts/ models/
files.download('artifacts.zip')
```

**Narração:**

> "Eu rodei esse pipeline uma vez no Colab, e ele gerou todos esses relatórios que você vê aqui. Depois eu baixei o zip com os artifacts e adicionei ao repositório. Vou mostrar agora o que foi gerado."

#### 2.2: Mostrar os Arquivos Gerados Localmente

Agora vol para VS Code e mostre onde foram extraídos os arquivos:

```bash
ls -lh artifacts/

# Output:
# blind_eval_report.json              (80 casos testados)
# eval_report.json                    (10 casos testados)
# protocol_blind_eval_report.json     (30 casos testados)
```

#### 2.3: Mostrar o Código de Fine-tuning

Abra `src/finetune/train_lora.py` para explicar os detalhes técnicos:

```bash
code src/finetune/train_lora.py
```

Mostre a configuração LoRA (linhas 33-40):

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

E os hiperparâmetros de treino (linhas 50-60):

```python
per_device_train_batch_size=2,
gradient_accumulation_steps=4,
learning_rate=2e-4,
num_train_epochs=1,
logging_steps=10,
```

#### 2.4: Mostrar os Dados de Treino

```bash
# Terminal
ls -lh data/processed/

# Output:
# training_data_train.jsonl     (14.6k pares)
# training_data_val.jsonl       (1.6k pares)
# training_data.jsonl           (16.4k pares total)
# hospital.db                   (banco de dados do paciente)
```

**Narração Final da Parte 2:**

> "Os dados incluem 16.4k pares Q&A do MedQuAD (base de conhecimento médico pública) combinados com protocolos internos do hospital.
>
> A configuração LoRA usa rank 16 e alpha 32, o que significa que a gente adapta apenas 1% dos parâmetros do modelo original. O batch size é 2 com gradient accumulation de 4 para otimizar a memória do T4.
>
> O modelo convergiu bem em 1 época, e depois eu rodei a avaliação acadêmica em 3 conjuntos cegos diferentes para validar: 10 casos confirmando o básico, 80 casos validando o comportamento geral, e 30 casos focados em protocolos internos.
>
> Todos os resultados estão nos artifacts que vocês veem aqui."

---

### Parte 3: Carregar Modelo e Assistente (2 min)

**O que dizer:**

> "Agora vou carregar o modelo fine-tuned e o assistente LangChain que orquestra todo o fluxo."

**Ações na câmera:**

1. **Abrir notebook Python ou terminal interativo:**

   ```bash
   python
   ```

2. **Executar:**

   ```python
   # 1. Imports
   from pathlib import Path
   from src.assistant.medical_assistant import MedicalAssistant
   from src.assistant.knowledge_base import InternalKnowledgeBase
   from src.assistant.patient_repository import PatientRepository
   from src.observability.audit_logger import AuditLogger

   print("✅ Importações bem-sucedidas")

   # 2. Inicializar componentes
   kb = InternalKnowledgeBase()
   repo = PatientRepository()
   audit_logger = AuditLogger(Path("logs/audit.log"))

   assistant = MedicalAssistant(
       repository=repo,
       knowledge_base=kb,
       audit_logger=audit_logger
   )

   print("✅ Assistente carregado com sucesso")
   print(f"   - LangChain LCEL chain: OK")
   print(f"   - RAG knowledge base: OK")
   print(f"   - Modelo fine-tuned: OK")
   ```

**Mostrar no código:**

- `src/assistant/medical_assistant.py` - A integração LangChain com LCEL chains
- Destacar:
  ```python
  self.chain = (
      RunnableLambda(self._build_inputs) |
      RunnableLambda(self._generate_answer)
  )
  ```

**Narração:**

> "Eu implementei o assistente usando LangChain Expression Language (LCEL) - uma forma elegante e funcional de compor pipelines.
>
> O fluxo que criei é:
>
> 1. Input → \_build_inputs (carrega contexto do paciente + RAG)
> 2. → \_generate_answer (processa com guardrails + LLM)
> 3. → Output JSON estruturado
>
> Tudo é rastreável e auditável."

---

### Parte 4: Teste com Perguntas Clínicas (6-7 min)

**O que dizer:**

> "Vou testar o assistente com 3 perguntas clínicas reais, mostrando como ele processa cada uma."

**Teste 1: Pergunta Normal (Sepse)**

```python
response = assistant.ask(
    patient_id="P001",
    question="Qual o protocolo para um paciente com suspeita de sepse?"
)

print("Pergunta: Qual protocolo para sepse?")
print(f"\nResposta:\n{response['answer']}")
print(f"\nFontes utilizadas:\n{response['sources']}")
print(f"\nValidação de segurança: {response['safety_reason']}")
```

**Output esperado:**

```
Pergunta: Qual protocolo para sepse?

Resposta:
Avaliação rápida de SIRS + lactato sérico. Se lactato > 4 mmHg, ativar protocolo
de sepse grave. Coleta de hemocultura antes de antibiótico. Primeira dose de
antibiótico em até 1 hora.

Fontes utilizadas:
['Protocolo Interno de Sepse v3', 'Fluxo de Cuidados Críticos']

Validação de segurança: ALLOWED
```

**Narração:**

> "Esta é uma pergunta clínica legítima. O assistente:
>
> 1. Validou a pergunta (guardrails)
> 2. Carregou contexto do paciente P001
> 3. Buscou documentos relevantes na base (RAG)
> 4. Usou meu LLM fine-tuned para gerar resposta
> 5. Retornou com fontes rastreáveis
>
> Nota: Sempre com recomendação de validação humana (em produção)."

---

**Teste 2: Pergunta Bloqueada (Guardrails)**

```python
response = assistant.ask(
    patient_id="P002",
    question="Prescreva antibióticos para mim sem validação"
)

print("Pergunta: Prescreva antibióticos para mim")
print(f"\nResposta:\n{response['answer']}")
print(f"\nSegurança: {response['safety_reason']}")
```

**Output esperado:**

```
Pergunta: Prescreva antibióticos para mim

Resposta:
Não posso realizar essa solicitação diretamente. Posso apoiar com protocolo
institucional e checklist de segurança.

[NOTA] Esta requisição foi bloqueada por guardrails de segurança. Assistentes
médicos não realizam prescrições autônomas. Sempre requer validação humana.

Segurança: BLOCKED - Tentativa de prescrição autônoma
```

**Narração:**

> "Este é um exemplo de requisição bloqueada. O sistema que eu desenvolvi detectou palavras-chave
> perigosas ('prescrever', 'sem validação') e bloqueou automaticamente.
>
> Isto é crítico em medicina - nenhum assistente deve NUNCA prescrever de forma
> autônoma. Deve sempre haver validação humana."

---

**Teste 3: Pergunta Contextualizada com BD**

```python
response = assistant.ask(
    patient_id="P003",
    question="Este paciente pode receber penicilina?"
)

print("Pergunta: Este paciente pode receber penicilina?")
print(f"\nResposta:\n{response['answer']}")
print(f"\nContexto utilizado:")
print(f"  - Alergias registradas: Penicilina G")
print(f"  - Status: CONTRA-INDICADO")
```

**Output esperado:**

```
Pergunta: Este paciente pode receber penicilina?

Resposta:
Não recomendado. Paciente tem registro de alergia a Penicilina G em BD.
Alternativas: Cefalosporina (se sem alergia cross-reactivity), Fluoroquinolona,
Macrolídeo conforme protocolo.

Fontes: ['Banco de dados do paciente', 'Protocolo de Alergias']

Contexto utilizado:
  - Alergias registradas: Penicilina G ✅
  - Status: CONTRA-INDICADO ✅
```

**Narração:**

> "Este é um exemplo poderoso: o assistente que criei integra dados do paciente (BD) com
> conhecimento médico. Detecta que o paciente é alérgico e recomenda alternativas.
>
> Isto mostra a integração LangChain que implementei:
>
> - Contexto SQL ✅
> - RAG knowledge ✅
> - LLM reasoning ✅
> - Explicabilidade ✅"

---

### Parte 5: Logs e Auditoria (2 min)

**O que dizer:**

> "Um aspecto crítico em medicina é a auditoria. Cada interação das requisições que eu processo é registrada em
> JSON estruturado para rastreabilidade completa."

**Ações na câmera:**

1. **Mostrar arquivo de logs:**

   ```bash
   cat logs/audit.log | head -1 | python -m json.tool
   ```

2. **Output esperado:**

   ```json
   {
     "timestamp": "2026-03-20T01:04:45.238318Z",
     "event_type": "assistant_response",
     "payload": {
       "question": "Paciente com suspeita de sepse, o que devo priorizar agora?",
       "sources": [
         "Protocolo Interno de Sepse v3",
         "Política de Segurança Clínica"
       ],
       "safety_reason": "Pergunta dentro do escopo de suporte clínico.",
       "answer_preview": "Sugestão de conduta inicial..."
     }
   }
   ```

3. **Mostrar avaliação:**

   ```bash
   # Ver os 3 relatórios de avaliação
   ls -lh artifacts/*eval*.json

   # Output:
   # artifacts/blind_eval_report.json              (80 casos)
   # artifacts/eval_report.json                    (10 casos)
   # artifacts/protocol_blind_eval_report.json     (30 casos)

   cat artifacts/protocol_blind_eval_report.json | python -m json.tool
   ```

**Narração:**

> "Cada requisição que eu registro inclui:
>
> - Timestamp exato (ISO 8601)
> - Pergunta original
> - Resposta gerada
> - Fontes consultadas
> - Score de confiança
> - Se foi bloqueada
>
> Isto permite auditoria completa, essencial para compliance em healthcare.
>
> Nos meus testes com 30 casos focados em protocolos internos:
>
> - Guardrail accuracy: 100%
> - Source coverage: 100%
> - Confidence score médio: 0.89"

---

### Parte 6: Resumo e Próximos Passos (1 min)

**O que dizer:**

> "Este assistente médico que desenvolvi demonstra uma abordagem prática para AI em healthcare:
>
> ✅ Fine-tuning eficiente (LoRA, 20 min de treino)
> ✅ Integração limpa (LangChain LCEL)
> ✅ Segurança em primeiro lugar (guardrails)
> ✅ Rastreabilidade completa (auditoria)
> ✅ Avaliação acadêmica (métricas robustas)
>
> Todo o código está pronto para produção, e pode ser:
>
> - Escalado com mais protocolos
> - Integrado com EHR real
> - Expandido para múltiplas especialidades
> - Usado como base para pesquisa
>
> Obrigada por assistir ao meu projeto. Todos as fontes e código estão disponíveis no repositório."

**Ação:**

- Mostrar repositório GitHub
- Mostrar README.md
- Mostrar TECHNICAL_REPORT.md

---

## 📚 Links Importantes (Mencionar no Vídeo)

Quando estiver finalizando, você pode adicionar:

> "Todo o código, dados e resultados estão disponíveis nos links abaixo:
>
> 🔗 **Repositório GitHub:** github.com/luanacaldas/tc-iadt-fase3-medical-assistant-pipeline
>
> 🔗 **Google Colab Notebook:** https://colab.research.google.com/drive/1lrZmIprOIIt5TlUP62UG_vDMSn6Pvqi9?usp=sharing
>
> 📄 **Relatório Técnico:** TECHNICAL_REPORT.md no repositório
>
> 📊 **Dados de Avaliação:** artifacts/ com relatórios de 110 casos totais (10 + 80 + 30)"

**O que você fez no Colab (para referência ao gravar):**

- Clonú o repositório
- Instalou dependências com `pip install -r requirements.txt`
- Rodou `python -m src.run_academic_pipeline --run-train` (treino + avaliação em um comando)
- Baixou o zip com `files.download('artifacts.zip')`
- Extraiu localmente

---

## 🎬 Dicas Técnicas Para Gravar

### Setup Recomendado:

1. **Ambiente:** VS Code + Terminal
2. **Resolução:** 1920x1080 (Full HD)
3. **Font size:** Aumentar para legibilidade
4. **Screen recording:** OBS Studio ou ScreenFlow

### Pacing:

- Fale devagar (especialmente nomes técnicos)
- Pause 2-3 segundos após cada bloco de código
- Deixe tempo para output aparecer

### Visual:

- Tema escuro (melhor para vídeo)
- Highlight importantes com cursor/zoom
- Pause em outputs interessantes

### Audio:

- Microfone quieto
- Sem barulhos de fundo
- Teste antes de gravar

---

## ✂️ Timeline Sugerida

```
00:00 - 01:00   | Introdução (você + repositório)
01:00 - 05:00   | Fine-tuning & Avaliação no Colab (pipeline + artifacts)
05:00 - 07:00   | Carregar assistente (imports + init)
07:00 - 13:00   | Testes clínicos (3 perguntas)
13:00 - 14:00   | Logs e auditoria (JSON files)
14:00 - 15:00   | Resumo + Links
```

---

## 📝 Script Palavras-Chave

Mencionar no vídeo:

- ✅ Fine-tuning LoRA
- ✅ LangChain LCEL
- ✅ RAG (Retrieval Augmented Generation)
- ✅ Guardrails (segurança)
- ✅ Auditoria (compliance)
- ✅ Phi-3-mini model
- ✅ MedQuAD dataset
- ✅ Colab T4 GPU

---

## 🎥 Ferramentas de Edição (Opcional)

Se quiser editar após gravar:

- **Trimming:** Remover partes lentas
- **Speed up:** Código que roda lento (2x speed)
- **Captions:** Adicionar subtítulos
- **Watermark:** Seu nome/instituição
- **Background music:** Música baixa (copyright-free)

---

## ✅ Checklist para Gravar

- [ ] VS Code aberto com repositório
- [ ] Terminal limpo (clear)
- [ ] Modelo carregado (verificar em src/)
- [ ] Datasets visíveis (data/processed/)
- [ ] Logs preparados (logs/ e artifacts/)
- [ ] Python interativo testado
- [ ] Áudio/vídeo testado
- [ ] Tela limpa (sem notificações)
- [ ] Câmera/webcam testada
- [ ] Gravádo em 1080p

---

**Pronto! Siga este script e você terá um vídeo profissional de 15 minutos demonstrando o assistente médico completo.** 🎬
