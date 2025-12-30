"""
Supervised Fine-Tuning (SFT) training.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from data.preference_datasets import load_preference_data
from utils import load_yaml


def main():
    # Load configs
    model_cfg = load_yaml("configs/model.yaml")
    data_cfg = load_yaml("configs/data.yaml")
    sft_cfg = load_yaml("configs/sft.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_length = model_cfg["max_length"]

    # Load preference data
    preference_data = load_preference_data(data_cfg)
    # fine tune on chosen responses only
    sft_data = [
        {
            "prompt": ex["prompt"],
            "response": ex["chosen"],
        }
        for ex in preference_data
    ]

    dataloader = DataLoader(
        sft_data,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: batch,
    )

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name"]).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = AdamW(
        model.parameters(),
        lr=sft_cfg["learning_rate"],
        weight_decay=sft_cfg.get("weight_decay", 0.0),
    )

    # Training loop
    model.train()

    for epoch in range(sft_cfg["epochs"]):
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"SFT Epoch {epoch + 1}/{sft_cfg['epochs']}",
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            prompts = [ex["prompt"] for ex in batch]
            responses = [ex["response"] for ex in batch]

            # Tokenize prompt + response with fixed context window
            inputs = tokenizer(
                prompts,
                responses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"SFT Epoch {epoch + 1} | Avg loss: {avg_loss:.4f}")

    # Save reference policy
    output_dir = sft_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved SFT (reference) model to {output_dir}")


if __name__ == "__main__":
    main()
