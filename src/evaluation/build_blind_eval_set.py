from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BLOCK_TERMS = [
    "prescreva",
    "prescricao",
    "prescrição",
    "dose exata",
    "receita",
]


def _is_validation_question(question: str, validation_ratio: float) -> bool:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < validation_ratio


def _guess_expected_source(question: str) -> str | None:
    q = question.lower()
    if any(term in q for term in ["sepse", "lactato", "hemocultura", "hipotens"]):
        return "Sepse"
    if any(term in q for term in ["torác", "torac", "troponina", "ecg", "supra"]):
        return "Torácica"
    if any(term in q for term in ["prescre", "receita", "dose exata", "validação", "validacao"]):
        return "Segurança"
    return None


def _iter_questions(paths: list[Path]) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                record = json.loads(line)
                question = str(record.get("input", "")).strip()
                if not question:
                    continue
                normalized = question.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                questions.append(question)

    return questions


def build_blind_eval_set(
    raw_paths: list[Path],
    output_path: Path,
    validation_ratio: float = 0.02,
    max_cases: int = 80,
    focused_per_source: int = 10,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = _iter_questions(raw_paths)
    candidate_questions = [q for q in questions if _is_validation_question(q, validation_ratio)]

    focused_buckets: dict[str, list[str]] = {
        "Sepse": [],
        "Torácica": [],
        "Segurança": [],
    }
    generic_questions: list[str] = []

    for question in candidate_questions:
        expected = _guess_expected_source(question)
        if expected in focused_buckets:
            focused_buckets[expected].append(question)
        else:
            generic_questions.append(question)

    selected_questions: list[str] = []
    for source_name in ["Sepse", "Torácica", "Segurança"]:
        selected_questions.extend(focused_buckets[source_name][:focused_per_source])

    remaining = max_cases - len(selected_questions)
    if remaining > 0:
        selected_questions.extend(generic_questions[:remaining])

    blind_questions = selected_questions[:max_cases]

    with output_path.open("w", encoding="utf-8") as outfile:
        for index, question in enumerate(blind_questions, start=1):
            lower = question.lower()
            should_block = any(term in lower for term in BLOCK_TERMS)
            expected = _guess_expected_source(question)
            record: dict[str, object] = {
                "id": f"B{index:04d}",
                "patient_id": "P001",
                "question": question,
                "should_block": should_block,
            }
            if expected:
                record["expected_source_contains"] = expected
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(blind_questions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria conjunto de avaliação cega a partir de perguntas reais.")
    parser.add_argument("--validation-ratio", type=float, default=0.02, help="Fração de perguntas reservada para avaliação cega.")
    parser.add_argument("--max-cases", type=int, default=80, help="Máximo de casos no conjunto cego.")
    parser.add_argument(
        "--focused-per-source",
        type=int,
        default=10,
        help="Quantidade alvo por tema com expectativa de fonte (Sepse/Torácica/Segurança).",
    )
    parser.add_argument("--output", default="data/eval/blind_eval_set.jsonl", help="Arquivo de saída do conjunto cego.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = [
        Path("data/raw/internal_clinical_qa.jsonl"),
        Path("data/raw/medquad_qa.jsonl"),
    ]
    total = build_blind_eval_set(
        raw_paths=paths,
        output_path=Path(args.output),
        validation_ratio=args.validation_ratio,
        max_cases=args.max_cases,
        focused_per_source=args.focused_per_source,
    )
    print(f"Conjunto cego criado com {total} casos em {args.output}")
