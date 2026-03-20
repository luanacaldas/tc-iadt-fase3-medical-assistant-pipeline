from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyResult:
    allowed: bool
    reason: str
    response_suffix: str


class Guardrails:
    blocked_terms = [
        "prescreva",
        "prescrever",
        "prescricao",
        "prescrição",
        "prescrição direta",
        "dose exata",
        "receita",
        "conduta definitiva",
        "conduta terapêutica final",
        "esquema medicamentoso definitivo",
        "autônoma",
        "autonoma",
        "sem validação",
        "sem revisao",
        "sem revisão",
        "sem intervenção",
        "sem intervencao",
        "assine virtualmente",
        "autorize prescrição",
        "autorize prescricao",
    ]

    def evaluate(self, user_question: str) -> SafetyResult:
        lower = user_question.lower()
        if any(term in lower for term in self.blocked_terms):
            return SafetyResult(
                allowed=False,
                reason="Tentativa de prescrição direta sem validação humana.",
                response_suffix=(
                    "\n\n⚠️ Limite de segurança: não forneço prescrição direta. "
                    "Encaminhe para validação do médico responsável."
                ),
            )

        return SafetyResult(
            allowed=True,
            reason="Pergunta dentro do escopo de suporte clínico.",
            response_suffix="\n\nValidação médica obrigatória antes de qualquer conduta definitiva.",
        )
