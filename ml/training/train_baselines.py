import numpy as np         
import pandas as pd          # DataFrames 
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
from sklearn.pipeline import Pipeline                 # Combine steps

# ==========================================
#    Scikit-Learn: Models 
# ==========================================
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR                              
from sklearn.neural_network import MLPRegressor          
from sklearn.neighbors import KNeighborsRegressor         

# ==========================================
#      Basic Config 
# ==========================================
LOGGER = logging.getLogger("train_baselines")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path("runtime/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RANDOM_STATE = 42

# ==========================================
#  Data Splitting & Cleaning
# ==========================================
def split_features_target(df: pd.DataFrame, target_col: str = "target", test_size=0.2, validation_size=0.2):
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not present")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # --- [CRITICAL FIX] ---
    X = X.select_dtypes(include=[np.number])

    total_rows = len(df)
    test_split_idx = int(total_rows * (1 - test_size))

    X_temp, X_test = X.iloc[:test_split_idx], X.iloc[test_split_idx:]
    y_temp, y_test = y[:test_split_idx], y[test_split_idx:]

    validation_split_idx = int(len(X_temp) * (1 - validation_size))

    X_train, X_validation = X_temp.iloc[:validation_split_idx], X_temp.iloc[validation_split_idx:]
    y_train, y_validation = y_temp[:validation_split_idx], y_temp[validation_split_idx:]

    LOGGER.info(f"Split data into: Train={len(X_train)}, Validation={len(X_validation)}, Test={len(X_test)}")
    
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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])



def build_models(random_state : int = DEFAULT_RANDOM_STATE) -> dict:
    models = {
        "hgb": make_pipelines(HistGradientBoostingRegressor(random_state=random_state)),
        "svr": make_pipelines(SVR()),
        "mlp": make_pipelines(MLPRegressor(hidden_layer_sizes=(64,32), random_state=random_state, max_iter=500)),
        "knn": make_pipelines(KNeighborsRegressor(n_neighbors=5))
    }
    return models

# ==========================================
#  Training and Evaluation
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
        # استخدام string كـ key للـ JSON لضمان التوافق
        calib_dict[str(alpha)] = float(quantile)
        LOGGER.info(f"Conformal Quantile for alpha={alpha}: {quantile:.4f}")
    return calib_dict

# ==========================================
#  The Orchestrator
# ==========================================
def train_pipeline(processed_csv_path: Path) -> dict:

    project_name = processed_csv_path.stem
    output_dir = MODEL_DIR / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Starting training pipeline for: {project_name} ...")
    
    
    df = pd.read_csv(processed_csv_path)
    splits = split_features_target(df, target_col="target")
    
    
    models = build_models()
    
    trained_models = {}
    for name, mdl in models.items():
        LOGGER.info(f"Training {name} ...")
        mdl.fit(splits["X_train"], splits["y_train"])
        trained_models[name] = mdl

        joblib.dump(mdl, output_dir / f"{name}.joblib")

    val_results = evaluate(trained_models, splits["X_validation"], splits["y_validation"])
    weights = compute_weights(val_results)
    
    # (Calibration)
    ensemble_val_pred = ensemble_predict(trained_models, weights, splits["X_validation"])    
    calib_dict = calibrate_conformal(splits["y_validation"], ensemble_val_pred)    

    # التقييم النهائي
    ensemble_test_pred = ensemble_predict(trained_models, weights, splits["X_test"])
    final_test_mae = mean_absolute_error(splits["y_test"], ensemble_test_pred)
    
    
    metadata = {
        "Weights": weights,
        "val_results": val_results,
        "final_test_mae": float(final_test_mae),
        "Calibration": calib_dict,
        "features_used": list(splits["X_train"].columns)
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=4))
    
    return metadata

# ==========================================
# 11. Standalone Test (Smoke Test)
# ==========================================

def run_standalone_test():
    """
    بيخلق داتا وهمية وبيشغل الـ Pipeline بالكامل للتأكد إن كل الأجزاء متربطة صح.
    """
    LOGGER.info("🛠️ Running Standalone Smoke Test...")
    
    np.random.seed(42)
    data_size = 1000
    test_df = pd.DataFrame({
        'irradiance': np.random.uniform(0, 1000, data_size),
        'ambient_temp': np.random.uniform(20, 45, data_size),
        'module_temp': np.random.uniform(20, 60, data_size),
        'hour_sin': np.random.uniform(-1, 1, data_size),
        'hour_cos': np.random.uniform(-1, 1, data_size),
        'target': np.random.uniform(0, 100, data_size) 
    })
    
    test_csv = Path("runtime/test_dummy_data.csv")
    test_df.to_csv(test_csv, index=False)
    
    try:
        metrics = train_pipeline(test_csv)
        
        LOGGER.info("✅ STANDALONE TEST SUCCESSFUL!")
        LOGGER.info(f"Final Test MAE: {metrics['final_test_mae']:.4f}")
        
        expected_files = ["hgb.joblib", "svr.joblib", "ensemble_metadata_test_dummy_data.json"]
        for f in expected_files:
            if (MODEL_DIR / f).exists():
                LOGGER.info(f"✔ Found generated file: {f}")
            else:
                LOGGER.warning(f"❌ Missing file: {f}")
                
    except Exception as e:
        LOGGER.error(f"💥 Standalone Test Failed: {e}")
    finally:
      
        if test_csv.exists():
            test_csv.unlink()

if __name__ == "__main__":
    run_standalone_test()
