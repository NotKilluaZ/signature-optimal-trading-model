from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIGS_DIR = ROOT_DIR / "configs"

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFESTS_DATA_DIR = DATA_DIR / "manifests"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

RUNS_DIR = ROOT_DIR / "runs"

# Function to create direcotries if they do not exist or get deleted
def ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MANIFESTS_DATA_DIR, SYNTHETIC_DATA_DIR, RUNS_DIR]:
        path.mkdir(parents = True, exist_ok = True)

# Return raw data directory for specified provider
def get_raw_provider_dir(provider: str) -> Path:
    return RAW_DATA_DIR / provider
