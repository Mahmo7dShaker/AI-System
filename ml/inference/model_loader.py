import joblib
import json
from pathlib import Path
import logging

LOGGER = logging.getLogger("inference")

def load_model(model_dir: Path):
    models = {}
    for name in ["hgb", "svr", "mlp", "knn"]:
        model_path = model_dir / f"{name}.joblib"
        if model_path.exists():
            LOGGER.info(f"Loading model {name} from {model_path}")
            models[name] = joblib.load(model_path)
        else:
            LOGGER.warning(f"Model  {model_path} not found. Skipping {name}.")

    meta_path = model_dir / "metadata.json"
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        LOGGER.info(f"Loaded metadata from {meta_path}")

    return models, metadata
