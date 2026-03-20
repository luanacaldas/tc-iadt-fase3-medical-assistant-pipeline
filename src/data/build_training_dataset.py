from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from src.data.preprocess import curate_record


def _iter_jsonl_records(paths: Iterable[Path]) -> Iterable[tuple[dict[str, str], str]]:
    for path in paths:
        if not path.exists():
            continue
        origin = path.stem
        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                yield json.loads(line), origin


def _is_validation_record(question: str, validation_ratio: float) -> bool:
    if validation_ratio <= 0:
        return False
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < validation_ratio


def build_instruction_dataset(
    raw_paths: list[Path],
    output_path: Path,
    internal_multiplier: int = 6,
    medquad_multiplier: int = 1,
    validation_ratio: float = 0.0,
    train_output_path: Path | None = None,
    validation_output_path: Path | None = None,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if validation_ratio > 0:
        train_path = train_output_path or Path("data/processed/training_data_train.jsonl")
        val_path = validation_output_path or Path("data/processed/training_data_val.jsonl")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.parent.mkdir(parents=True, exist_ok=True)

        train_written = 0
        val_written = 0

        with train_path.open("w", encoding="utf-8") as train_file, val_path.open("w", encoding="utf-8") as val_file:
            for raw_record, origin in _iter_jsonl_records(raw_paths):
                record = curate_record(raw_record)
                prompt = (
                    "Você é um assistente médico institucional. Use protocolos internos e indique limites clínicos.\n"
                    f"Pergunta: {record['input']}\n"
                    "Resposta:"
                )
                model_record = {
                    "text": f"{prompt} {record['output']}",
                    "source": record["source"],
                    "domain": record["domain"],
                }

                is_val = _is_validation_record(record["input"], validation_ratio)
                if is_val:
                    val_file.write(json.dumps(model_record, ensure_ascii=False) + "\n")
                    val_written += 1
                    continue

                multiplier = medquad_multiplier if "medquad" in origin.lower() else internal_multiplier
                for _ in range(max(multiplier, 1)):
                    train_file.write(json.dumps(model_record, ensure_ascii=False) + "\n")
                    train_written += 1

        return {
            "train_written": train_written,
            "val_written": val_written,
            "total_written": train_written + val_written,
        }

    written = 0
    with output_path.open("w", encoding="utf-8") as outfile:
        for raw_record, origin in _iter_jsonl_records(raw_paths):
            record = curate_record(raw_record)
            prompt = (
                "Você é um assistente médico institucional. Use protocolos internos e indique limites clínicos.\n"
                f"Pergunta: {record['input']}\n"
                "Resposta:"
            )
            model_record = {
                "text": f"{prompt} {record['output']}",
                "source": record["source"],
                "domain": record["domain"],
            }

            multiplier = medquad_multiplier if "medquad" in origin.lower() else internal_multiplier
            for _ in range(max(multiplier, 1)):
                outfile.write(json.dumps(model_record, ensure_ascii=False) + "\n")
                written += 1

    return {
        "train_written": written,
        "val_written": 0,
        "total_written": written,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dataset final de treino combinando fontes clínicas.")
    parser.add_argument(
        "--internal-multiplier",
        type=int,
        default=6,
        help="Fator de oversampling para dados internos do hospital.",
    )
    parser.add_argument(
        "--medquad-multiplier",
        type=int,
        default=1,
        help="Fator de amostragem para dados MedQuAD.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/training_data.jsonl",
        help="Arquivo final de treino em JSONL.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help="Percentual (0 a 1) reservado para validação cega sem oversampling.",
    )
    parser.add_argument(
        "--output-train",
        default="data/processed/training_data_train.jsonl",
        help="Arquivo de saída para split de treino quando --validation-ratio > 0.",
    )
    parser.add_argument(
        "--output-val",
        default="data/processed/training_data_val.jsonl",
        help="Arquivo de saída para split de validação quando --validation-ratio > 0.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_sources = [
        Path("data/raw/internal_clinical_qa.jsonl"),
        Path("data/raw/medquad_qa.jsonl"),
    ]
    processed = Path(args.output)
    stats = build_instruction_dataset(
        raw_sources,
        processed,
        internal_multiplier=args.internal_multiplier,
        medquad_multiplier=args.medquad_multiplier,
        validation_ratio=args.validation_ratio,
        train_output_path=Path(args.output_train),
        validation_output_path=Path(args.output_val),
    )
    if args.validation_ratio > 0:
        print(
            "Split gerado com "
            f"train={stats['train_written']} e val={stats['val_written']} "
            f"(ratio={args.validation_ratio}, internal_multiplier={args.internal_multiplier}, "
            f"medquad_multiplier={args.medquad_multiplier})"
        )
    else:
        print(
            "Dataset combinado gerado com "
            f"{stats['total_written']} exemplos em {processed} "
            f"(internal_multiplier={args.internal_multiplier}, medquad_multiplier={args.medquad_multiplier})"
        )
