import os
import sys
import yaml
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llm_agent import LLMAgent
from scripts.utils_distill import Distiller

def prepare_chatml_format(example):
    system_prompt = "You are a sophisticated financial analyst. Your task is to analyze a natural language instruction about a market scenario and break it down into a list of mutually exclusive potential outcomes."
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nInstruction: {example['instruction']}\nJSON Output:<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return {"text": text}

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_data = pd.read_pickle(os.path.join(config['data']['processed_path'], "dataset_v1.pkl"))
    distiller = Distiller(api_key=os.getenv("OPENAI_API_KEY"))
    distilled_list = distiller.batch_distill(raw_data, limit=2000)

    raw_ds = Dataset.from_list(distilled_list)
    dataset = raw_ds.map(prepare_chatml_format, remove_columns=raw_ds.column_names)

    agent = LLMAgent(config['llm']['model_name'], use_lora=True, use_4bit=True)
    model = prepare_model_for_kbit_training(agent.model)

    training_args = TrainingArguments(
        output_dir=config['llm']['output_dir'],
        num_train_epochs=config['llm']['num_epochs'],
        per_device_train_batch_size=config['llm']['batch_size'],
        gradient_accumulation_steps=4,
        learning_rate=float(config['llm']['learning_rate']),
        bf16=True, 
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=agent.tokenizer,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(agent.tokenizer, padding=True)
    )

    trainer.train()
    model.save_pretrained(os.path.join(config['llm']['output_dir'], "final_adapter"))
    print("LLM SFT logic aligned and training completed.")

if __name__ == "__main__":
    main()