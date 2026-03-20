import re
from typing import Any


PII_PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "EMAIL": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    "PHONE": re.compile(r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b"),
}


def anonymize_text(text: str) -> str:
    sanitized = text
    for label, pattern in PII_PATTERNS.items():
        sanitized = pattern.sub(f"[{label}_REDACTED]", sanitized)
    return sanitized


def curate_record(record: dict[str, Any]) -> dict[str, Any]:
    required_fields = ["input", "output", "source"]
    missing = [field for field in required_fields if field not in record or not record[field]]
    if missing:
        raise ValueError(f"Registro inválido. Campos ausentes: {missing}")

    return {
        "input": anonymize_text(str(record["input"])),
        "output": anonymize_text(str(record["output"])),
        "source": str(record["source"]),
        "domain": str(record.get("domain", "general")),
    }
