from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments


@dataclass
class FineTuneConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path: Path = Path("data/processed/training_data.jsonl")
    output_dir: Path = Path("artifacts/phi3-medical-lora")
    max_length: int = 1024


def train_lora(config: FineTuneConfig) -> None:
    dataset = load_dataset("json", data_files=str(config.dataset_path), split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


if __name__ == "__main__":
    train_lora(FineTuneConfig())
