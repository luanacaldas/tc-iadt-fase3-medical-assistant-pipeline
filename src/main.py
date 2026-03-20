from src.assistant.knowledge_base import InternalKnowledgeBase
from src.assistant.medical_assistant import MedicalAssistant
from src.assistant.patient_repository import PatientRepository
from src.assistant.workflow import ClinicalWorkflow
from src.config import get_config
from src.observability.audit_logger import AuditLogger


def run_demo() -> None:
    config = get_config()

    repository = PatientRepository(config.db_path)
    repository.seed_demo_data()

    audit_logger = AuditLogger(config.audit_log_path)
    knowledge_base = InternalKnowledgeBase()
    assistant = MedicalAssistant(repository, knowledge_base, audit_logger)
    workflow = ClinicalWorkflow(repository, assistant)

    question = "Paciente com suspeita de sepse, o que devo priorizar agora?"
    result = workflow.run(patient_id="P001", question=question)

    print("=== RESPOSTA DO ASSISTENTE ===")
    print(result["assistant"]["answer"])
    print("\n=== FONTES ===")
    for source in result["assistant"]["sources"]:
        print(f"- {source}")
    print("\n=== ALERTAS ===")
    for alert in result["alerts"]:
        print(f"- {alert}")
    print("\n=== SEGURANÇA ===")
    print(result["assistant"]["safety_reason"])


if __name__ == "__main__":
    run_demo()
