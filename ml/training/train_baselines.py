

import numpy as np         
import pandas as pd          # DataFrames (جداول البيانات زي الإكسيل)
import json                 # Save metadata in JSON format
# ==========================================
#    System & Files
# ==========================================
import joblib                # Save/Load models 
from pathlib import Path
import logging               # Track events & errors

# ==========================================
#   Scikit-Learn: Prep & Eval 
# ==========================================
from sklearn.model_selection import train_test_split  # Split data and evaluate
from sklearn.metrics import mean_absolute_error, r2_score # KPIs 
from sklearn.preprocessing import StandardScaler      # Scale numbers in SVR and MLP
from sklearn.pipeline import Pipeline                 # Combine steps (دمج التظبيط والتدريب في خطوة واحدة)

# ==========================================
#    Scikit-Learn: Models 
# ==========================================
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR                              
from sklearn.neural_network import MLPRegressor          
from sklearn.neighbors import KNeighborsRegressor         
# ==========================================
#      Basic Config (الإعدادات الأساسية)
# ==========================================
# Setup logging
LOGGER = logging.getLogger("train_baselines")
logging.basicConfig(level=logging.INFO)

# Define models directory 
MODEL_DIR = Path("runtime/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Fix random state
DEFAULT_RANDOM_STATE = 42

def split_features_target(df: pd.DataFrame, target_col: str = "target", test_size=0.2, validation_size=0.2):
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not present")
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False) 

    validation_rel = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation =train_test_split(X_temp, y_temp, test_size=validation_rel, shuffle=False)

    return {
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train,
        "X_validation": X_validation.reset_index(drop=True),
        "y_validation": y_validation,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test
    }

# ==========================================
# model Pipelines
# ==========================================
def make_pipelines(model):
    
    if isinstance(model, HistGradientBoostingRegressor):
        return model
    
    return Pipeline([
        ("scaler", StandardScaler()),  
        ("model", model) 
    ])

def build_models(random_state : int = DEFAULT_RANDOM_STATE) -> dict:
    models = {
        "hgb": make_pipelines(HistGradientBoostingRegressor(random_state=random_state)),
        "svr": make_pipelines(SVR()),
        "mlp": make_pipelines(MLPRegressor(hidden_layer_sizes=(64,32), random_state=random_state, max_iter=500)),        "knn": make_pipelines(KNeighborsRegressor(n_neighbors=5))
    }
    return models

# ==========================================
# 5. traning and Evaluation
# ==========================================
def train_single_models(models: dict, X_train : pd.DataFrame, y_train: np.ndarray) -> dict:
    trained = {}
    for name, mdl in models.items():
        LOGGER.info(f"training {name} ...")

        mdl.fit(X_train, y_train)
        trained[name] = mdl
        
        joblib.dump(mdl, MODEL_DIR / f"{name}.joblib")

    return trained
    
def evaluate(models: dict, X_validation: pd.DataFrame, y_validation: np.ndarray) -> dict:
    results = {}
    for name, mdl in models.items():
        LOGGER.info(f"evaluating {name} ...")
        y_pred = mdl.predict(X_validation)
        mae = mean_absolute_error(y_validation, y_pred)
        results[name] = mae
        LOGGER.info(f"{name} MAE: {mae:.4f}")
    return results

# ==========================================
# Ensemble & Dynamic Weighting
# ==========================================
def compute_weights(mae_dict: dict, epsilon = 1e-8) -> dict:
    inv_performance = {k : 1.0 / (v + epsilon) for k, v in mae_dict.items()}
    total_inv = sum(inv_performance.values())
    weights = {k: float(inv_performance[k] / total_inv) for k in inv_performance.keys()}

    return weights

def ensemble_predict(models: dict, weights: dict, X: pd.DataFrame) -> np.ndarray:   
    # cinvert data to matrix 2D
    preds = []
    for name, mdl in models.items():
        preds.append(mdl.predict(X))
    
    preds_matrix = np.vstack(preds)
    model_keys = list(models.keys())
    weights_array = np.array([weights.get(k, 0.0) for k in model_keys])

    ensemble_prediction = np.average(preds_matrix, axis=0, weights=weights_array)
    return ensemble_prediction


# ==========================================
# Conformal Calibration
# ==========================================
def calibrate_conformal(y_val: np.ndarray, y_val_pred: np.ndarray, alphas: list = [0.1, 0.5]) -> dict:
    residuals = np.abs(y_val - y_val_pred)
    calib_dict = {}
    for alpha in alphas:
        quantile = np.quantile(residuals, 1 - alpha)
        calib_dict[alpha] = float(quantile)
        LOGGER.info(f"Conformal Quantile for alpha={alpha}: {quantile:.4f}")
    return calib_dict
# ==========================================
# main function training pipeline
# ==========================================
def train_pipeline(processed_csv_path: Path) -> dict:
    
    LOGGER.info(f"Starting training pipeline for: {processed_csv_path.name} ...")
    # lood data
    df = pd.read_csv(processed_csv_path)
    splits = split_features_target(df, target_col="target")
    
    models = build_models()
    trained_models = train_single_models(models, splits["X_train"], splits["y_train"])

    val_results = evaluate(trained_models, splits["X_validation"], splits["y_validation"])
    weights = compute_weights(val_results)
    LOGGER.info(f"Model Weights: {weights}")

    ensemble_val_pred = ensemble_predict(trained_models, weights, splits["X_validation"])    
    calib_dict = calibrate_conformal(splits["y_validation"], ensemble_val_pred)    

    # Finad Unbiased testing
    ensemble_test_pred = ensemble_predict(trained_models, weights, splits["X_test"])
    final_test_mae = mean_absolute_error(splits["y_test"], ensemble_test_pred)
    LOGGER.info(f"Final Ensemble Test MAE: {final_test_mae:.4f}")

    # Matadata Saving
    metadata = {
        "Weights": weights,
        "val_results": val_results,
        "final_test_mae": float(final_test_mae),
        "Calibration": calib_dict
    }

    metadata_path = MODEL_DIR / f"ensemble_metadata_{processed_csv_path.stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=4))
    LOGGER.info(f"Saved Matadata to {metadata_path}")

    return metadata