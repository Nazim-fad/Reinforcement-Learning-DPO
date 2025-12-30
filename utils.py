import yaml


def load_yaml(path):
    """Load a YAML file that contains configuration settings."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
