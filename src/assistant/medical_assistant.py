from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.assistant.knowledge_base import InternalKnowledgeBase
from src.assistant.patient_repository import PatientRepository
from src.observability.audit_logger import AuditLogger
from src.security.guardrails import Guardrails


class MedicalAssistant:
    def __init__(self, repository: PatientRepository, knowledge_base: InternalKnowledgeBase, audit_logger: AuditLogger) -> None:
        self.repository = repository
        self.knowledge_base = knowledge_base
        self.audit_logger = audit_logger
        self.guardrails = Guardrails()

        self.prompt = ChatPromptTemplate.from_template(
            """
            Você é um assistente médico institucional.
            Responda de forma objetiva, não prescreva diretamente e sempre peça validação humana.

            Contexto do paciente:
            {patient_context}

            Exames pendentes:
            {pending_exams}

            Trechos de protocolos internos:
            {retrieved_snippets}

            Pergunta do médico:
            {question}
            """
        )

        self.chain = RunnableLambda(self._build_inputs) | RunnableLambda(self._generate_answer)

    def _build_inputs(self, payload: dict[str, str]) -> dict[str, Any]:
        patient_context = self.repository.get_patient_context(payload["patient_id"])
        docs = self.knowledge_base.retrieve(payload["question"])

        snippets = [doc.text for doc in docs]
        sources = [doc.source for doc in docs]

        return {
            "question": payload["question"],
            "patient_context": patient_context,
            "pending_exams": patient_context.get("pending_exams", []),
            "retrieved_snippets": snippets,
            "sources": sources,
        }

    def _generate_answer(self, context: dict[str, Any]) -> dict[str, Any]:
        safety = self.guardrails.evaluate(context["question"])

        if not safety.allowed:
            response = (
                "Não posso realizar essa solicitação diretamente. "
                "Posso apoiar com protocolo institucional e checklist de segurança."
                + safety.response_suffix
            )
        else:
            prompt_value = self.prompt.invoke(
                {
                    "patient_context": context["patient_context"],
                    "pending_exams": context["pending_exams"],
                    "retrieved_snippets": context["retrieved_snippets"],
                    "question": context["question"],
                }
            )
            prompt_text = "\n".join(message.content for message in prompt_value.messages)
            response = (
                "Sugestão de conduta inicial: priorizar estabilização, revisar exames pendentes "
                "e aplicar protocolo institucional compatível com quadro clínico.\n\n"
                f"Resumo contextual:\n{prompt_text[:700]}"
                + safety.response_suffix
            )

        output = {
            "answer": response,
            "sources": context["sources"],
            "safety_reason": safety.reason,
        }

        self.audit_logger.log_event(
            "assistant_response",
            {
                "question": context["question"],
                "sources": context["sources"],
                "safety_reason": safety.reason,
                "answer_preview": response[:300],
            },
        )

        return output

    def ask(self, patient_id: str, question: str) -> dict[str, Any]:
        return self.chain.invoke({"patient_id": patient_id, "question": question})
