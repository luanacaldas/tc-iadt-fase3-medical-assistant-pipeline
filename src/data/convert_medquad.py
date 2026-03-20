from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from datasets import load_dataset


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return _clean_text("".join(node.itertext()))


def _pick_first(row: dict[str, object], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = _clean_text(str(value))
        if text:
            return text
    return default


def _iter_medquad_records(medquad_root: Path) -> tuple[list[dict[str, str]], int]:
    records: list[dict[str, str]] = []
    skipped = 0

    for xml_path in medquad_root.rglob("*.xml"):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            skipped += 1
            continue

        source = _clean_text(root.attrib.get("source", "MedQuAD"))
        focus = _clean_text(root.findtext("Focus"))
        document_url = _clean_text(root.attrib.get("url", ""))

        qa_pairs = root.find("QAPairs")
        if qa_pairs is None:
            skipped += 1
            continue

        for qa_pair in qa_pairs.findall("QAPair"):
            question_node = qa_pair.find("Question")
            answer_node = qa_pair.find("Answer")

            question = _node_text(question_node)
            answer = _node_text(answer_node)
            qtype = _clean_text(question_node.attrib.get("qtype", "general") if question_node is not None else "general")
            qid = _clean_text(question_node.attrib.get("qid", "") if question_node is not None else "")

            if not question or not answer:
                skipped += 1
                continue

            if answer.lower() in {"topics", "see answer", "n/a"}:
                skipped += 1
                continue

            records.append(
                {
                    "input": question,
                    "output": answer,
                    "source": f"MedQuAD-{source}",
                    "domain": "medquad",
                    "qtype": qtype,
                    "focus": focus,
                    "qid": qid,
                    "document_url": document_url,
                }
            )

    return records, skipped


def convert_medquad_to_jsonl(medquad_root: Path, output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records, skipped = _iter_medquad_records(medquad_root)

    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "written": len(records),
        "skipped": skipped,
    }


def convert_medquad_hf_to_jsonl(output_path: Path, hf_dataset_name: str = "lavita/MedQuAD") -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(hf_dataset_name, split="train")

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as outfile:
        for row in dataset:
            question = _pick_first(row, ["question", "Question", "query"])
            answer = _pick_first(row, ["answer", "Answer", "response"])

            if not question or not answer:
                skipped += 1
                continue

            if answer.lower() in {"topics", "see answer", "n/a"}:
                skipped += 1
                continue

            source = _pick_first(row, ["source", "Source"], default="MedQuAD")
            qtype = _pick_first(row, ["qtype", "question_type"], default="general")
            focus = _pick_first(row, ["focus", "question_focus"], default="")
            qid = _pick_first(row, ["qid", "id"], default="")
            document_url = _pick_first(row, ["url", "document_url", "source_url"], default="")

            record = {
                "input": question,
                "output": answer,
                "source": f"MedQuAD-{source}",
                "domain": "medquad",
                "qtype": qtype,
                "focus": focus,
                "qid": qid,
                "document_url": document_url,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {
        "written": written,
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converte MedQuAD para JSONL padronizado.")
    parser.add_argument(
        "--source",
        choices=["xml", "hf"],
        default="xml",
        help="Fonte de dados: xml local (MedQuAD-master) ou Hugging Face (lavita/MedQuAD).",
    )
    parser.add_argument(
        "--medquad-dir",
        default="MedQuAD-master",
        help="Diretório raiz dos XMLs MedQuAD (quando --source xml).",
    )
    parser.add_argument(
        "--hf-dataset",
        default="lavita/MedQuAD",
        help="Nome do dataset no Hugging Face (quando --source hf).",
    )
    parser.add_argument(
        "--output",
        default="data/raw/medquad_qa.jsonl",
        help="Arquivo JSONL de saída.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.output)

    if args.source == "hf":
        stats = convert_medquad_hf_to_jsonl(out, hf_dataset_name=args.hf_dataset)
        print(f"MedQuAD (HF) convertido: {stats['written']} registros, {stats['skipped']} ignorados -> {out}")
    else:
        medquad_dir = Path(args.medquad_dir)
        stats = convert_medquad_to_jsonl(medquad_dir, out)
        print(f"MedQuAD (XML) convertido: {stats['written']} registros, {stats['skipped']} ignorados -> {out}")
