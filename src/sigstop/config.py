import yaml
from pathlib import Path
from src.sigstop.paths import CONFIGS_DIR

# Open the default YAML config file
def load_config(path: str | Path = CONFIGS_DIR / "default.yaml") -> dict:
    path = Path(path)
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)