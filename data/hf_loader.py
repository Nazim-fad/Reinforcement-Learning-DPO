"""
HF dataset loader for preference data
"""
from datasets import load_dataset


def load_from_hf(data_cfg):
    dataset_name = data_cfg["dataset_name"]
    split = data_cfg.get("split", "train")
    max_samples = data_cfg.get("max_samples")

    ds = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    preference_data = []

    for item in ds:
        try:
            preference_data.append({
                "prompt": item["input"], # specific to argilla/distilabel-intel-orca-dpo-pairs
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
        except KeyError as e:
            raise KeyError(
                f"HF dataset {dataset_name} is missing required field: {e}"
            )

    return preference_data
