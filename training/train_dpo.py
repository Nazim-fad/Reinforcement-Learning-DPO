"""
Training loop for DPO
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data.preference_datasets import PreferenceDataset
from DPO_Implementation.models.policy import Policy
from DPO_Implementation.losses.dpo import DPOLoss
from utils import load_yaml
from data.preference_datasets import load_preference_data


def main():
    # Config setup
    model_cfg = load_yaml("configs/model.yaml")
    data_cfg = load_yaml("configs/data.yaml")
    dpo_cfg = load_yaml("configs/dpo.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    preference_data = load_preference_data(data_cfg)

    dataset = PreferenceDataset(preference_data)
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: batch
    )

    # Init models
    policy = Policy(
        dpo_cfg["sft_checkpoint"],
        model_cfg["max_length"]
    ).to(device)
    reference = Policy(
        dpo_cfg["sft_checkpoint"],
        model_cfg["max_length"],
        trainable=False
    ).to(device)

    # Loss and optimizer
    dpo_loss = DPOLoss(beta=dpo_cfg["beta"])
    optimizer = AdamW(
        policy.parameters(),
        lr=dpo_cfg["learning_rate"]
    )

    # Training loop
    policy.train()

    for epoch in range(dpo_cfg["epochs"]):
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{dpo_cfg['epochs']}")

        for batch in progress_bar:
            optimizer.zero_grad()

            prompts = [ex["prompt"] for ex in batch]
            chosen = [ex["chosen"] for ex in batch]
            rejected = [ex["rejected"] for ex in batch]

            # Compute log Pi_theta
            log_pi_theta_chosen = torch.stack([
                policy.log_prob(x, y) for x, y in zip(prompts, chosen)
            ]).to(device)

            log_pi_theta_rejected = torch.stack([
                policy.log_prob(x, y) for x, y in zip(prompts, rejected)
            ]).to(device)

            # Compute log Pi_ref
            with torch.no_grad():
                log_pi_ref_chosen = torch.stack([
                    reference.log_prob(x, y) for x, y in zip(prompts, chosen)
                ]).to(device)

                log_pi_ref_rejected = torch.stack([
                    reference.log_prob(x, y) for x, y in zip(prompts, rejected)
                ]).to(device)

            # DPO loss
            loss = dpo_loss(
                log_pi_theta_chosen,
                log_pi_theta_rejected,
                log_pi_ref_chosen,
                log_pi_ref_rejected,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} | DPO loss: {avg_loss:.4f}")

    # Save trained policy
    output_dir = dpo_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    policy.model.save_pretrained(output_dir)
    policy.tokenizer.save_pretrained(output_dir)

    print(f"Saved DPO policy to {output_dir}")


if __name__ == "__main__":
    main()
