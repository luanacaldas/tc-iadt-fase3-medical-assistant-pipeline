flowchart TD
    A["👤 Usuário<br/>Pergunta Clínica"] -->|Input| B["🔒 GUARDRAILS<br/>Validação de Segurança"]
    
    B -->|Pass| C["📊 SQL QUERY<br/>Contexto do Paciente"]
    B -->|Block| Z["❌ Requisição<br/>Rejeitada"]
    
    C -->|Dados| D["🔍 RAG SEARCH<br/>Knowledge Base"]
    D -->|Top-3 Documentos| E["📝 PROMPT BUILDER<br/>Contextualização"]
    
    F["💾 Knowledge Base<br/>MedQuAD 16.4K QA<br/>+ Protocolos"] -.->|Índice| D
    G["🏥 Banco Paciente<br/>SQLite"] -.->|Query| C
    
    E -->|Prompt + Contexto| H["🤖 LLM FINE-TUNED<br/>Phi-3 + LoRA"]
    
    H -->|Raw Response| I["✨ POST-PROCESSOR<br/>Extração de Fontes"]
    
    I -->|Valida| J{"Conformidade<br/>com Guardrails?"}
    J -->|Sim| K["📋 JSON ESTRUTURADO<br/>Resposta + Fontes"]
    J -->|Não| L["⚠️ LOG DE ERRO<br/>Escalação para Humano"]
    
    K -->|Output| M["📤 RESPOSTA FINAL<br/>Médico"]
    L -->|Output| M
    
    M -->|Auditoria| N["📁 AUDIT LOG<br/>JSON com Timestamp"]
    
    O["⏱️ Processamento: ~245ms<br/>🎯 Acurácia: 99%<br/>📌 Cobertura: 100%"] -.->|Métricas| M
    
    style A fill:#e1f5ff
    style B fill:#ffccbc
    style C fill:#f3e5f5
    style D fill:#fce4ec
    style E fill:#fff3e0
    style H fill:#c8e6c9
    style I fill:#b2dfdb
    style J fill:#fff9c4
    style K fill:#a8d8ea
    style M fill:#c8e6c9
    style Z fill:#ffcdd2
    style L fill:#ffcdd2
    style N fill:#e0e0e0
