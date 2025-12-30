"""
Take Raw data turn it to a clean RL-style interface for DPO training
(prompt, chosen, rejected)
"""

from torch.utils.data import Dataset
from data.hf_loader import load_from_hf
from data.local_loader import load_from_local


class PreferenceDataset(Dataset):
    """
    Generic preference dataset for DPO.
    Each item is a (prompt, chosen, rejected) triple.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }


# Dispatch function to load data based on config
def load_preference_data(data_cfg):
    """
    Load preference data according to the source: "hf" or "local"
    """

    source = data_cfg["source"]

    if source == "hf":
        return load_from_hf(data_cfg)

    elif source == "local":
        return load_from_local(data_cfg)

    else:
        raise ValueError(
            f"Unknown data source '{source}'. Expected 'hf' or 'local'."
        )