from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass
class ProtocolDocument:
    source: str
    text: str


class InternalKnowledgeBase:
    def __init__(self) -> None:
        self.documents = [
            ProtocolDocument(
                source="Protocolo Interno de Sepse v3",
                text=(
                    "Paciente com suspeita de sepse deve ter lactato e culturas coletadas cedo, "
                    "antibiótico em até 1 hora e avaliação hemodinâmica seriada."
                ),
            ),
            ProtocolDocument(
                source="Fluxo Dor Torácica v2",
                text=(
                    "Dor torácica sem supra: realizar ECG seriado, repetir troponina em janela definida "
                    "e estratificar risco para conduta segura."
                ),
            ),
            ProtocolDocument(
                source="Política de Segurança Clínica",
                text=(
                    "Assistente virtual não realiza prescrição final. Toda sugestão depende de validação "
                    "humana por médico responsável."
                ),
            ),
        ]

        self.keyword_boosts = {
            "Protocolo Interno de Sepse v3": {
                "sepse",
                "septic",
                "choque",
                "lactato",
                "hemocultura",
                "hipotensao",
                "hipotensão",
                "febre",
                "antibiotico",
                "antibiótico",
                "primeira hora",
                "exames pendentes",
            },
            "Fluxo Dor Torácica v2": {
                "dor toracica",
                "dor torácica",
                "troponina",
                "ecg",
                "supra",
                "st",
                "isquemia",
                "estratificar risco",
            },
            "Política de Segurança Clínica": {
                "prescreva",
                "prescrever",
                "prescricao",
                "prescrição",
                "dose exata",
                "receita",
                "alta",
                "conduta definitiva",
                "conduta terapeutica final",
                "conduta terapêutica final",
                "esquema medicamentoso definitivo",
                "autonoma",
                "autônoma",
                "sem validacao",
                "sem validação",
                "sem revisao",
                "sem revisão",
                "sem intervencao",
                "sem intervenção",
                "assine virtualmente",
                "autorize prescricao",
                "autorize prescrição",
                "validacao",
                "validação",
            },
        }

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return " ".join(normalized.split())

    def _score(self, query: str, document: ProtocolDocument) -> int:
        normalized_query = self._normalize(query)
        normalized_text = self._normalize(document.text)

        query_tokens = set(normalized_query.split())
        text_tokens = set(normalized_text.split())
        overlap_score = len(query_tokens.intersection(text_tokens))

        boost_score = 0
        for keyword in self.keyword_boosts.get(document.source, set()):
            if self._normalize(keyword) in normalized_query:
                boost_score += 4

        return overlap_score + boost_score

    def retrieve(self, query: str, top_k: int = 2) -> list[ProtocolDocument]:
        ranked = sorted(self.documents, key=lambda doc: self._score(query, doc), reverse=True)
        return ranked[:top_k]
