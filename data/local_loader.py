"""
Local file (called train.json/csv/xlsx, etc.) for preference data
"""

import os
import json
import pandas as pd


def load_from_local(data_cfg):
    """
    Load preference data from a local folder CSV, JSON, or Excel file.
    Required fields: prompt, chosen, rejected
    """

    path = data_cfg["path"]
    split = data_cfg.get("split", "train")

    # supported file extensions
    extensions = ["json", "csv", "xlsx"]
    file_path = None

    for ext in extensions:
        candidate = os.path.join(path, f"{split}.{ext}")
        if os.path.exists(candidate):
            file_path = candidate
            break

    if file_path is None:
        raise FileNotFoundError(
            f"No data file found for split='{split}' in {path}. "
            f"Expected one of: {', '.join(extensions)}"
        )

    # Load data
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of examples.")

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")

    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        data = df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Validate schema
    required_fields = {"prompt", "chosen", "rejected"}
    preference_data = []

    for i, example in enumerate(data):
        if not required_fields.issubset(example):
            missing = required_fields - example.keys()
            raise KeyError(
                f"Example {i} is missing required fields: {missing}"
            )

        preference_data.append({
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        })

    return preference_data
