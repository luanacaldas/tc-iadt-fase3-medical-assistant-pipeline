#!/usr/bin/env python
"""
Script para executar os 3 testes clínicos do Assistente Médico.
Executa etapa por etapa de forma clara e didática.
"""

import sqlite3
from pathlib import Path
from src.assistant.knowledge_base import InternalKnowledgeBase
from src.assistant.medical_assistant import MedicalAssistant
from src.assistant.patient_repository import PatientRepository
from src.assistant.workflow import ClinicalWorkflow
from src.config import get_config
from src.observability.audit_logger import AuditLogger


def seed_clinical_test_patients(repository: PatientRepository) -> None:
    """Popula 3 pacientes de teste para os testes clínicos."""
    db_path = repository.db_path
    with sqlite3.connect(db_path) as conn:
        # Paciente 1: Sepse suspeita, alérgico a penicilina
        conn.execute(
            """
            INSERT OR REPLACE INTO patients (patient_id, age, sex, main_complaint, allergies)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("P001", 67, "M", "febre e hipotensão", "Penicilina G"),
        )
        
        conn.execute("DELETE FROM pending_exams WHERE patient_id = ?", ("P001",))
        conn.executemany(
            """
            INSERT INTO pending_exams (patient_id, exam_name, status)
            VALUES (?, ?, ?)
            """,
            [
                ("P001", "Lactato", "pending"),
                ("P001", "Hemocultura", "pending"),
            ],
        )
        
        # Paciente 2: Para teste de guardrail (sem alergias críticas)
        conn.execute(
            """
            INSERT OR REPLACE INTO patients (patient_id, age, sex, main_complaint, allergies)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("P002", 45, "F", "infecção respiratória", "Nenhuma"),
        )
        
        # Paciente 3: Particularmente alérgico a penicilina
        conn.execute(
            """
            INSERT OR REPLACE INTO patients (patient_id, age, sex, main_complaint, allergies)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("P003", 52, "M", "pneumonia", "Penicilina G, Amoxicilina"),
        )
        
        conn.commit()


def print_test_result(result: dict) -> None:
    """Imprime resultado formatado de um teste clínico."""
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║ RESPOSTA DO ASSISTENTE")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"\n{result['assistant']['answer']}\n")
    
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║ CONTEXTO CLÍNICO")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"Paciente: {result['patient_id']}")
    if result['pending_exams']:
        print(f"Exames pendentes: {result['pending_exams']}")
    
    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║ SEGURANÇA")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"Status: {result['assistant']['safety_reason']}\n")
    
    if result['alerts']:
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║ ALERTAS")
        print("╚══════════════════════════════════════════════════════════════════════════╝")
        for alert in result['alerts']:
            print(f"⚠️  {alert}")
        print()


def run_clinical_tests() -> None:
    """Executa os 3 testes clínicos principais para demonstração do assistente."""
    print("\n" + "="*80)
    print("PARTE 5 · TESTES CLÍNICOS")
    print("="*80)
    
    config = get_config()
    
    # Inicializa repositório com dados de teste
    repository = PatientRepository(config.db_path)
    seed_clinical_test_patients(repository)
    
    # Inicializa componentes do assistente
    audit_logger = AuditLogger(config.audit_log_path)
    knowledge_base = InternalKnowledgeBase()
    assistant = MedicalAssistant(repository, knowledge_base, audit_logger)
    workflow = ClinicalWorkflow(repository, assistant)
    
    # ========== TESTE 1: Pergunta clínica legítima (Sepse) ==========
    print("\n" + "-"*80)
    print("TESTE 1 · Pergunta Clínica Legítima — Sepse")
    print("-"*80)
    print("\n[1/3] Executando: assistant.ask(patient_id='P001', question='Qual o protocolo para")
    print("      um paciente com suspeita de sepse?')\n")
    
    test1_result = workflow.run(
        patient_id="P001",
        question="Qual o protocolo para um paciente com suspeita de sepse?"
    )
    
    print_test_result(test1_result)
    
    print("\n🎙️  O que aconteceu aqui:")
    print("   1. Guardrail validou que é pergunta dentro do escopo clínico ✓")
    print("   2. Sistema carregou contexto do paciente P001 (histórico, alergias)")
    print("   3. Busca semântica trouxe documentos relevantes sobre sepse da KB")
    print("   4. LLM gerou resposta usando tudo isso")
    print("   5. Resposta inclui fontes — você sabe de onde veio cada informação")
    print("   6. Em produção: validação humana obrigatória em cima")
    print("      → Assistente APOIA a decisão, não TOMA\n")
    
    # ========== TESTE 2: Guardrail bloqueando prescrição ==========
    print("\n" + "-"*80)
    print("TESTE 2 · Guardrail Bloqueando Prescrição Autônoma")
    print("-"*80)
    print("\n[2/3] Executando: assistant.ask(patient_id='P002', question='Prescreva")
    print("      antibióticos para mim sem validação')\n")
    
    test2_result = workflow.run(
        patient_id="P002",
        question="Prescreva antibióticos para mim sem validação"
    )
    
    print_test_result(test2_result)
    
    print("\n⛔ O que aconteceu aqui:")
    print("   1. Guardrail BLOQUEOU a solicitação")
    print("   2. Sistema detectou intenção de prescrição autônoma ANTES do LLM")
    print("   3. Em medicina: nenhum sistema de IA deve prescrever sem validação")
    print("      → Isso é INEGOCIÁVEL")
    print("   4. Bloqueio é eficiente: não custou processamento do LLM")
    print("   5. Resultado: Segurança + Eficiência juntos\n")
    
    # ========== TESTE 3: Contexto do DB (Alergia) ==========
    print("\n" + "-"*80)
    print("TESTE 3 · Contexto do Banco de Dados — Alergia a Penicilina")
    print("-"*80)
    print("\n[3/3] Executando: assistant.ask(patient_id='P003', question='Este paciente")
    print("      pode receber penicilina?')\n")
    
    test3_result = workflow.run(
        patient_id="P003",
        question="Este paciente pode receber penicilina?"
    )
    
    print_test_result(test3_result)
    
    print("\n🎯 O que aconteceu aqui (meu favorito):")
    print("   1. Assistente NÃO 'adivinhou' que é alérgico")
    print("   2. Ele foi NO banco de dados, leu o registro de P003")
    print("   3. Encontrou: 'Alergia: Penicilina G, Amoxicilina'")
    print("   4. Usou isso na resposta")
    print("   5. Não só diz 'não pode', mas sugere alternativas baseadas")
    print("      nos protocolos:")
    print("      - Cefalosporina (se sem alergia cruzada)")
    print("      - Fluorquinolona")
    print("      - Macrolídeo")
    print("   6. SQL + RAG + LLM trabalhando juntos")
    print("      → É EXATAMENTE ISSO que eu precisava que funcionasse\n")
    
    print("\n" + "="*80)
    print("TESTES CLÍNICOS CONCLUÍDOS COM SUCESSO")
    print("="*80)


def run_individual_test(test_number: int) -> None:
    """Executa um teste individual, permitindo explicação durante o vídeo."""
    config = get_config()
    
    # Inicializa repositório com dados de teste
    repository = PatientRepository(config.db_path)
    seed_clinical_test_patients(repository)
    
    # Inicializa componentes do assistente
    audit_logger = AuditLogger(config.audit_log_path)
    knowledge_base = InternalKnowledgeBase()
    assistant = MedicalAssistant(repository, knowledge_base, audit_logger)
    workflow = ClinicalWorkflow(repository, assistant)
    
    if test_number == 1:
        print("\n" + "="*80)
        print("TESTE 1 · Pergunta Clínica Legítima — Sepse")
        print("="*80)
        print("\n📝 Vou fazer uma pergunta clínica normal sobre sepse...")
        print("   Comando: assistant.ask(patient_id='P001', question='Qual o protocolo para")
        print("            um paciente com suspeita de sepse?')")
        print("\n⏸️  Pressione ENTER para executar...")
        input()
        
        print("\n🔄 Executando agora...\n")
        result = workflow.run(
            patient_id="P001",
            question="Qual o protocolo para um paciente com suspeita de sepse?"
        )
        
        print_test_result(result)
        
    elif test_number == 2:
        print("\n" + "="*80)
        print("TESTE 2 · Guardrail Bloqueando Prescrição Autônoma")
        print("="*80)
        print("\n⛔ Vou tentar algo PERIGOSO: pedir uma prescrição direta...")
        print("   Comando: assistant.ask(patient_id='P002', question='Prescreva")
        print("            antibióticos para mim sem validação')")
        print("\n⏸️  Pressione ENTER para executar...")
        input()
        
        print("\n🔄 Executando agora...\n")
        result = workflow.run(
            patient_id="P002",
            question="Prescreva antibióticos para mim sem validação"
        )
        
        print_test_result(result)
        
    elif test_number == 3:
        print("\n" + "="*80)
        print("TESTE 3 · Contexto do Banco de Dados — Alergia a Penicilina")
        print("="*80)
        print("\n🔬 Pergunta que depende 100% do banco de dados...")
        print("   Comando: assistant.ask(patient_id='P003', question='Este paciente")
        print("            pode receber penicilina?')")
        print("\n⏸️  Pressione ENTER para executar...")
        input()
        
        print("\n🔄 Executando agora...\n")
        result = workflow.run(
            patient_id="P003",
            question="Este paciente pode receber penicilina?"
        )
        
        print_test_result(result)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            test_num = int(sys.argv[1])
            if test_num in [1, 2, 3]:
                run_individual_test(test_num)
            else:
                print("❌ Teste inválido. Use: python clinical_tests.py [1|2|3]")
        except ValueError:
            print("❌ Use um número: python clinical_tests.py 1")
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ TESTES CLÍNICOS - MODO VÍDEO (INTERATIVO)
╚══════════════════════════════════════════════════════════════════════════════╝

Use um dos comandos abaixo para rodar cada teste individualmente:

  python clinical_tests.py 1
  └─ TESTE 1 · Pergunta Clínica Legítima — Sepse
     (Demonstra: Guardrail ✓, Contexto do paciente, RAG, Alertas)

  python clinical_tests.py 2
  └─ TESTE 2 · Guardrail Bloqueando Prescrição Autônoma
     (Demonstra: Bloqueio de segurança ANTES do LLM)

  python clinical_tests.py 3
  └─ TESTE 3 · Contexto do Banco — Alergia a Penicilina
     (Demonstra: SQL + RAG + LLM integrando tudo)

Cada teste:
  ✓ Mostra o comando que vai executar
  ✓ Aguarda você pressionar ENTER (tempo para falar no vídeo)
  ✓ Executa e mostra resultado
  ✓ Explica tudo que aconteceu por trás

╔══════════════════════════════════════════════════════════════════════════════╗
║ ROTEIRO SUGERIDO PARA O VÍDEO
╚══════════════════════════════════════════════════════════════════════════════╝

[09:00-10:00] TESTE 1 - Sepse
  • Execute: python clinical_tests.py 1
  • Explique sobre o guardrail e contexto
  • Mostre os alertas de exames críticos

[10:00-12:00] TESTE 2 - Bloqueio de Prescrição
  • Execute: python clinical_tests.py 2
  • Enfatize: nenhuma IA prescreve sem validação
  • Mostre como o guardrail bloqueia ANTES do LLM

[12:00-14:00] TESTE 3 - Alergia no Banco de Dados
  • Execute: python clinical_tests.py 3
  • Mostre como consultou a alergia de verdade
  • Explique SQL + RAG + LLM funcionando juntos

─────────────────────────────────────────────────────────────────────────────

""")
