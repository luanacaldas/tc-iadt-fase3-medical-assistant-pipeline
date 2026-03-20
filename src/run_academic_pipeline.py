from __future__ import annotations

import argparse
from pathlib import Path

from src.data.build_training_dataset import build_instruction_dataset
from src.data.convert_medquad import convert_medquad_hf_to_jsonl
from src.data.convert_medquad import convert_medquad_to_jsonl
from src.evaluation.build_blind_eval_set import build_blind_eval_set
from src.evaluation.build_eval_set import build_eval_set
from src.evaluation.build_protocol_blind_set import build_protocol_blind_set
from src.evaluation.evaluate_assistant import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa pipeline acadêmico ponta a ponta em um comando.")
    parser.add_argument("--medquad-source", choices=["xml", "hf"], default="xml")
    parser.add_argument("--medquad-dir", default="MedQuAD-master")
    parser.add_argument("--medquad-hf-dataset", default="lavita/MedQuAD")
    parser.add_argument("--internal-multiplier", type=int, default=8)
    parser.add_argument("--medquad-multiplier", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--run-train", action="store_true", help="Executa fine-tuning LoRA (demora e requer recursos).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    medquad_output = Path("data/raw/medquad_qa.jsonl")
    if args.medquad_source == "hf":
        medquad_stats = convert_medquad_hf_to_jsonl(medquad_output, hf_dataset_name=args.medquad_hf_dataset)
        print(f"[1/6] MedQuAD (HF) convertido: {medquad_stats['written']} registros")
    else:
        medquad_stats = convert_medquad_to_jsonl(Path(args.medquad_dir), medquad_output)
        print(f"[1/6] MedQuAD (XML) convertido: {medquad_stats['written']} registros")

    dataset_stats = build_instruction_dataset(
        raw_paths=[Path("data/raw/internal_clinical_qa.jsonl"), Path("data/raw/medquad_qa.jsonl")],
        output_path=Path("data/processed/training_data.jsonl"),
        internal_multiplier=args.internal_multiplier,
        medquad_multiplier=args.medquad_multiplier,
        validation_ratio=args.validation_ratio,
        train_output_path=Path("data/processed/training_data_train.jsonl"),
        validation_output_path=Path("data/processed/training_data_val.jsonl"),
    )
    print(
        "[2/6] Split dataset gerado: "
        f"train={dataset_stats['train_written']} val={dataset_stats['val_written']}"
    )

    if args.run_train:
        from src.finetune.train_lora import FineTuneConfig
        from src.finetune.train_lora import train_lora

        print("[3/6] Iniciando fine-tuning LoRA...")
        train_lora(FineTuneConfig(dataset_path=Path("data/processed/training_data_train.jsonl")))
        print("[3/6] Fine-tuning concluído")
    else:
        print("[3/6] Fine-tuning pulado (use --run-train para ativar)")

    build_eval_set(Path("data/eval/academic_eval_set.jsonl"))
    academic_metrics = evaluate(Path("data/eval/academic_eval_set.jsonl"), Path("artifacts/eval_report.json"))
    print(f"[4/6] Avaliação acadêmica concluída: {academic_metrics}")

    build_blind_eval_set(
        raw_paths=[Path("data/raw/internal_clinical_qa.jsonl"), Path("data/raw/medquad_qa.jsonl")],
        output_path=Path("data/eval/blind_eval_set.jsonl"),
        validation_ratio=0.2,
        max_cases=80,
        focused_per_source=10,
    )
    blind_metrics = evaluate(Path("data/eval/blind_eval_set.jsonl"), Path("artifacts/blind_eval_report.json"))
    print(f"[5/6] Avaliação cega concluída: {blind_metrics}")

    build_protocol_blind_set(
        output_path=Path("data/eval/protocol_blind_eval_set.jsonl"),
        internal_raw_path=Path("data/raw/internal_clinical_qa.jsonl"),
    )
    protocol_metrics = evaluate(
        Path("data/eval/protocol_blind_eval_set.jsonl"),
        Path("artifacts/protocol_blind_eval_report.json"),
    )
    print(f"[6/6] Blind Protocol Set concluído: {protocol_metrics}")

    print("Pipeline acadêmico finalizado com sucesso.")


if __name__ == "__main__":
    main()
