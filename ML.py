"""
Direct PID Predictor
====================

This file replaces the old surrogate approach:

    [car parameters + PID gains] -> predicted error

with the direct supervised-learning approach:

    [car parameters] -> optimal PID gains

The expensive optimiser should be run offline in MLPID_main.py to create a
training dataset of optimal gains for many different cars. This file then trains
models that predict those optimal gains directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Column definitions
# -----------------------------------------------------------------------------

CAR_PARAM_COLS = [
    "mass",
    "max_acceleration",
    "yaw_inertia",
    "length",
    "max_steering_stiffness_front",
    "max_steering_stiffness_rear",
    "tyre_friction",
    "cd",
    "cross_sectional_area",
]

STEER_TARGET_COLS = [
    "opt_steer_kp",
    "opt_steer_ki",
    "opt_steer_kd",
]

SPEED_TARGET_COLS = [
    "opt_speed_kp",
    "opt_speed_ki",
    "opt_speed_kd",
]

ALL_TARGET_COLS = STEER_TARGET_COLS + SPEED_TARGET_COLS


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def add_car_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically meaningful car-only derived features.

    Important:
    These features must NOT include PID gains. The model's job is to learn
    PID gains from car parameters only.
    """
    X = X.copy()

    X["aero_drag_proxy"] = X["cd"] * X["cross_sectional_area"]
    X["steer_balance"] = (
        X["max_steering_stiffness_front"]
        / (X["max_steering_stiffness_rear"] + 1e-9)
    )
    X["lateral_grip_front"] = (
        X["tyre_friction"] * X["max_steering_stiffness_front"]
    )
    X["lateral_grip_rear"] = (
        X["tyre_friction"] * X["max_steering_stiffness_rear"]
    )
    X["accel_per_mass"] = X["max_acceleration"] / (X["mass"] + 1e-9)
    X["yaw_factor"] = X["yaw_inertia"] / (
        X["mass"] * X["length"] ** 2 + 1e-9
    )
    X["stiffness_per_mass"] = (
        X["max_steering_stiffness_front"] + X["max_steering_stiffness_rear"]
    ) / (X["mass"] + 1e-9)
    X["grip_force"] = X["tyre_friction"] * X["mass"] * 9.81
    X["drag_to_mass"] = X["aero_drag_proxy"] / (X["mass"] + 1e-9)

    return X


# -----------------------------------------------------------------------------
# Direct PID model
# -----------------------------------------------------------------------------

@dataclass
class DirectPIDModel:
    """
    Multi-output model that predicts either:

        car parameters -> steering PID

    or:

        car parameters -> speed PID
    """

    target_cols: list[str]
    model: Pipeline | None = None
    feature_names: list[str] | None = None

    def _make_model(self, n_samples: int) -> Pipeline:
        if n_samples < 150:
            xgb_params = dict(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=42,
                n_jobs=-1,
            )
        elif n_samples < 500:
            xgb_params = dict(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
            )
        else:
            xgb_params = dict(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.2,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))),
        ])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "DirectPIDModel":
        X_feat = add_car_derived_features(X)
        self.feature_names = list(X_feat.columns)
        self.model = self._make_model(len(X_feat))
        self.model.fit(X_feat[self.feature_names], y[self.target_cols])
        return self

    def predict(self, car_params: dict | pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model has not been fitted or loaded.")

        if isinstance(car_params, dict):
            X = pd.DataFrame([car_params])
        else:
            X = car_params.copy()

        missing = [c for c in CAR_PARAM_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing car parameter columns: {missing}")

        X_feat = add_car_derived_features(X)
        pred = self.model.predict(X_feat[self.feature_names])
        pred = np.maximum(pred, 0.0)  # PID gains should not be negative

        return pd.DataFrame(pred, columns=self.target_cols)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        preds = self.predict(X)

        metrics = {}
        print("\nModel evaluation:")
        for col in self.target_cols:
            mae = mean_absolute_error(y[col], preds[col])
            r2 = r2_score(y[col], preds[col])
            metrics[col] = {"mae": mae, "r2": r2}
            print(f"  {col:<16} MAE = {mae:.5f} | R² = {r2:.4f}")

        return metrics

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"Model saved -> {path}")

    @staticmethod
    def load(path: str) -> "DirectPIDModel":
        model = joblib.load(path)
        print(f"Model loaded <- {path}")
        return model


# -----------------------------------------------------------------------------
# Data loading and training helpers
# -----------------------------------------------------------------------------

def load_optimal_pid_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='latin-1')

    required = CAR_PARAM_COLS + ALL_TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def train_direct_pid_models(
    csv_path: str,
    steer_model_path: str = "direct_steer_pid_model.pkl",
    speed_model_path: str = "direct_speed_pid_model.pkl",
    test_size: float = 0.2,
) -> dict:
    """
    Train two direct models:

        car parameters -> steering PID
        car parameters -> speed PID
    """
    df = load_optimal_pid_dataset(csv_path)

    X = df[CAR_PARAM_COLS].copy()
    y_steer = df[STEER_TARGET_COLS].copy()
    y_speed = df[SPEED_TARGET_COLS].copy()

    X_train, X_test, ys_train, ys_test, yv_train, yv_test = train_test_split(
        X,
        y_steer,
        y_speed,
        test_size=test_size,
        random_state=42,
    )

    print("=" * 60)
    print("Training direct steering PID model")
    print("=" * 60)
    steer_model = DirectPIDModel(target_cols=STEER_TARGET_COLS)
    steer_model.fit(X_train, ys_train)
    steer_metrics = steer_model.evaluate(X_test, ys_test)
    steer_model.save(steer_model_path)

    print("\n" + "=" * 60)
    print("Training direct speed PID model")
    print("=" * 60)
    speed_model = DirectPIDModel(target_cols=SPEED_TARGET_COLS)
    speed_model.fit(X_train, yv_train)
    speed_metrics = speed_model.evaluate(X_test, yv_test)
    speed_model.save(speed_model_path)

    return {
        "steer_model": steer_model,
        "speed_model": speed_model,
        "steer_metrics": steer_metrics,
        "speed_metrics": speed_metrics,
    }


def predict_optimal_pid(
    car_params: dict,
    steer_model_path: str = "direct_steer_pid_model.pkl",
    speed_model_path: str = "direct_speed_pid_model.pkl",
) -> dict:
    """
    Fast runtime inference.

    This does NOT run simulation or optimisation. It simply predicts PID gains
    from car parameters using the offline-trained models.
    """
    steer_model = DirectPIDModel.load(steer_model_path)
    speed_model = DirectPIDModel.load(speed_model_path)

    steer_pred = steer_model.predict(car_params).iloc[0].to_dict()
    speed_pred = speed_model.predict(car_params).iloc[0].to_dict()

    return {
        "steer_kp": float(steer_pred["opt_steer_kp"]),
        "steer_ki": float(steer_pred["opt_steer_ki"]),
        "steer_kd": float(steer_pred["opt_steer_kd"]),
        "speed_kp": float(speed_pred["opt_speed_kp"]),
        "speed_ki": float(speed_pred["opt_speed_ki"]),
        "speed_kd": float(speed_pred["opt_speed_kd"]),
    }


if __name__ == "__main__":
    train_direct_pid_models(
        csv_path = os.path.join(os.path.dirname(__file__), 'First150Good.csv'),
        steer_model_path="direct_steer_pid_model.pkl",
        speed_model_path="direct_speed_pid_model.pkl",
    )
