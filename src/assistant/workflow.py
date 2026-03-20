from __future__ import annotations

from typing import Any

from src.assistant.medical_assistant import MedicalAssistant
from src.assistant.patient_repository import PatientRepository


class ClinicalWorkflow:
    def __init__(self, repository: PatientRepository, assistant: MedicalAssistant) -> None:
        self.repository = repository
        self.assistant = assistant

    def run(self, patient_id: str, question: str) -> dict[str, Any]:
        context = self.repository.get_patient_context(patient_id)
        pending_exams = context.get("pending_exams", [])

        alerts = []
        critical_exams = {"Lactato", "Hemocultura"}
        missing_critical = [exam["exam_name"] for exam in pending_exams if exam["exam_name"] in critical_exams]

        if missing_critical:
            alerts.append(
                "Exames críticos pendentes identificados: " + ", ".join(missing_critical)
            )

        assistant_output = self.assistant.ask(patient_id=patient_id, question=question)

        return {
            "patient_id": patient_id,
            "pending_exams": pending_exams,
            "alerts": alerts,
            "assistant": assistant_output,
        }
