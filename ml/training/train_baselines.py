

import numpy as np         
import pandas as pd          # DataFrames (جداول البيانات زي الإكسيل)

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
    X_temp, X_test, ytemp, y_test = train_test_split(X, y, test_size=test_size, random_state=DEFAULT_RANDOM_STATE) 

    validation_rel = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation =train_test_split(X_temp, ytemp, test_size=validation_rel, random_state=DEFAULT_RANDOM_STATE)

    return {
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train,
        "X_validation": X_validation.reset_index(drop=True),
        "y_validation": y_validation,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test
    }

# ==========================================
# 5. traning and Evaluation
# ==========================================
def train_and_evaluate(models: dict, X_train : pd.DataFramr, y_train: np.ndarray) -> dict:
    trained = {}
    for nam, mdl in models.items():
        LOGGER.info(f"training {nam} ...")

        mdl.fit(X_train, y_train)
        trained[name] = mdl
        
        joblib.dump(mdl, MODEL_DIR / f' [name].jobllib')

        return trained
    
def evaluate(models: dict, X_validation: pd.DataFrame, y_validation: np.ndarray) -> dict:
    results = {}
    for name. mdl in models.items():
        LOGGER.info(f"evaluating {name} ...")
        y_pred = mdl.predict(X_validation)
        mae = mean_absolute_error(y_validation, y_pred)
        results[name] = mae
        LOGGER.info(f"{name} MAE: {mae:.4f}")
    return results

