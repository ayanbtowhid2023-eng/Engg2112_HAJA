"""
PID Coefficient Optimiser — Option B (Error-Aware) with Outlier Handling
=========================================================================
Stage 1: Clean extreme-error samples (unstable PID gains)
Stage 2: Train XGBoost surrogate:  [car_params + PID_values] -> error
Stage 3: At inference, run scipy optimiser to find PID values that
         minimise predicted error for a given set of car parameters.

Inputs (9):
    mass, max_acceleration, yaw_inertia, length,
    max_steering_stiffness_front, max_steering_stiffness_rear,
    tyre_friction, cd, cross_sectional_area

PID values (6, optimised at inference):
    steer_kp, steer_ki, steer_kd,
    speed_kp,  speed_ki,  speed_kd

Target (1, used only during training):
    simulation_error
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize, differential_evolution
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Column definitions ────────────────────────────────────────────────────────

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

PID_COLS = [
    "steer_kp", "steer_ki", "steer_kd",
    "speed_kp",  "speed_ki",  "speed_kd",
]

TARGET_COL = "simulation_error"
ALL_FEATURES = CAR_PARAM_COLS + PID_COLS  # 15 features fed to XGBoost


# ── Outlier handling ──────────────────────────────────────────────────────────

def recommend_cleaning_strategy(y: pd.Series) -> str:
    """
    Inspect the error distribution and recommend a cleaning strategy.
    Prints a short diagnostic report and returns the recommended method string.
    """
    n = len(y)
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = (y > Q3 + 3 * IQR).sum()
    outlier_pct   = 100 * outlier_count / n
    skew          = y.skew()

    print("\n── Error distribution diagnostic ─────────────────")
    print(f"  Samples              : {n}")
    print(f"  Mean / Median        : {y.mean():.3f} / {y.median():.3f}")
    print(f"  Std                  : {y.std():.3f}")
    print(f"  Skewness             : {skew:.2f}")
    print(f"  Outliers (>Q3+3*IQR) : {outlier_count} ({outlier_pct:.1f}%)")
    print(f"  Max error            : {y.max():.2f}")

    if n < 150:
        rec    = "cap"
        reason = "small dataset — dropping rows is costly, cap preserves all data"
    elif outlier_pct > 20:
        rec    = "log"
        reason = "very high outlier rate — log transform keeps data and compresses extremes"
    elif skew > 5:
        rec    = "log"
        reason = "heavily right-skewed — log transform normalises the target distribution"
    else:
        rec    = "iqr"
        reason = "standard case — IQR removal is clean and effective"

    print(f"\n  Recommended strategy : '{rec}'  ({reason})")
    return rec


def clean_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "iqr",
    iqr_multiplier: float = 3.0,
    percentile_cap: float = 99.0,
    abs_error_cap: float = None,
) -> tuple:
    """
    Remove or cap extreme simulation errors from unstable PID gains.

    Without cleaning, a handful of errors in the hundreds will dominate
    training and produce R² < 0 on the normal operating range.

    Args:
        method:
            'iqr'        — drop rows where error > Q3 + iqr_multiplier * IQR.
                           Best default for datasets with >150 clean samples.
            'percentile' — drop rows above percentile_cap (e.g. 99th).
                           Good when you have >500 samples.
            'cap'        — clip error to abs_error_cap (winsorisation).
                           Preserves row count — best for small datasets.
            'log'        — log1p-transform the target. Keeps all rows and
                           compresses extreme values. Predictions are
                           automatically expm1-inverted at inference time.
        iqr_multiplier:
            IQR sensitivity. Lower (1.5) = aggressive. Higher (5.0) = lenient.
        percentile_cap:
            Percentile threshold for 'percentile' method.
        abs_error_cap:
            Hard ceiling for 'cap' method. Defaults to 99th percentile.

    Returns:
        (X_clean, y_clean, transform_info)
        transform_info is passed to PIDSurrogateModel.fit() so predictions
        are correctly inverse-transformed at inference.
    """
    n_before = len(y)
    transform_info = {"log_transform": False, "cap_value": None}

    if method == "iqr":
        Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
        threshold = Q3 + iqr_multiplier * (Q3 - Q1)
        mask = y <= threshold
        X_out, y_out = X[mask].copy(), y[mask].copy()

    elif method == "percentile":
        threshold = y.quantile(percentile_cap / 100.0)
        mask = y <= threshold
        X_out, y_out = X[mask].copy(), y[mask].copy()

    elif method == "cap":
        cap = abs_error_cap if abs_error_cap is not None else float(y.quantile(0.99))
        transform_info["cap_value"] = cap
        X_out = X.copy()
        y_out = y.clip(upper=cap).copy()

    elif method == "log":
        transform_info["log_transform"] = True
        X_out = X.copy()
        y_out = np.log1p(y).copy()

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'iqr', 'percentile', 'cap', or 'log'.")

    n_after      = len(y_out)
    pct_removed  = 100 * (1 - n_after / n_before)

    print(f"\nOutlier cleaning  ({method}):")
    print(f"  Samples before   : {n_before}")
    print(f"  Samples after    : {n_after}  ({pct_removed:.1f}% removed)")
    print(f"  Error range now  : [{float(y_out.min()):.4f}, {float(y_out.max()):.4f}]")

    if n_after < 50:
        print(f"  WARNING: only {n_after} samples remain — consider 'cap' or 'log' instead.")

    return X_out.reset_index(drop=True), y_out.reset_index(drop=True), transform_info


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> tuple:
    """
    Load CSV dataset. Runs a cleaning recommendation diagnostic on load.

    Returns:
        X : DataFrame (n, 15) — car params + PID values (raw, uncleaned)
        y : Series    (n,)    — simulation error (raw, uncleaned)
    """
    df = pd.read_csv(filepath)
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    print(f"Loaded {len(df)} samples")
    return X, y


# ── Feature engineering ───────────────────────────────────────────────────────

def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Physics-motivated derived features."""
    X = X.copy()
    X["aero_drag_proxy"] = X["cd"] * X["cross_sectional_area"]
    X["steer_balance"]   = (X["max_steering_stiffness_front"]
                            / (X["max_steering_stiffness_rear"] + 1e-9))
    X["lateral_grip"]    = X["tyre_friction"] * X["max_steering_stiffness_front"]
    X["accel_per_mass"]  = X["max_acceleration"] / (X["mass"] + 1e-9)
    X["yaw_factor"]      = X["yaw_inertia"] / (X["mass"] * X["length"] ** 2 + 1e-9)
    if "steer_kp" in X.columns:
        X["steer_ki_kp_ratio"] = X["steer_ki"] / (X["steer_kp"] + 1e-9)
        X["speed_ki_kp_ratio"] = X["speed_ki"] / (X["speed_kp"] + 1e-9)
    return X


# ── Surrogate model ───────────────────────────────────────────────────────────

def _xgb_params_for_size(n: int) -> dict:
    """
    Scale model complexity to dataset size.
    Small datasets need shallower trees and stronger regularisation.

    Approximate minimum samples needed per complexity tier:
      <150  : shallow trees, heavy reg  (R² ~0.5–0.7 expected)
       150–400: medium depth            (R² ~0.75–0.85 expected)
       400+  : full complexity          (R² ~0.85–0.95 expected)
    """
    if n < 150:
        return dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=5.0)
    elif n < 400:
        return dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=2.0)
    else:
        return dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0)


class PIDSurrogateModel:
    """
    XGBoost surrogate: [car_params + PID_values] -> simulation_error.
    Automatically scales model complexity to dataset size and handles
    log-transform inversion at inference time.
    """

    def __init__(self):
        self.model          = None
        self.scaler         = StandardScaler()
        self.feature_names  = []
        self.pid_bounds     = {}
        self.car_param_bounds = {}
        self.transform_info = {"log_transform": False, "cap_value": None}

    def fit(self, X: pd.DataFrame, y: pd.Series,
            transform_info: dict = None, verbose: bool = True):

        if transform_info:
            self.transform_info = transform_info

        n = len(X)
        self.model = xgb.XGBRegressor(
            **_xgb_params_for_size(n),
            random_state=42, n_jobs=-1,
            early_stopping_rounds=30, eval_metric="mae",
        )

        for col in PID_COLS:
            self.pid_bounds[col] = (float(X[col].min()), float(X[col].max()))
        for col in CAR_PARAM_COLS:
            self.car_param_bounds[col] = (float(X[col].min()), float(X[col].max()))

        X_feat = add_derived_features(X)
        self.feature_names = list(X_feat.columns)
        X_scaled = self.scaler.fit_transform(X_feat.values)

        val_size = 0.15 if n < 200 else 0.2
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y.values, test_size=val_size, random_state=42
        )
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                       verbose=50 if verbose else False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feat  = add_derived_features(X)[self.feature_names]
        X_scaled = self.scaler.transform(X_feat.values)
        preds   = self.model.predict(X_scaled)
        if self.transform_info.get("log_transform"):
            preds = np.expm1(preds)
        return preds

    def evaluate(self, X: pd.DataFrame, y_raw: pd.Series) -> dict:
        """Always evaluate against raw (untransformed) error."""
        preds = self.predict(X)
        mae   = mean_absolute_error(y_raw, preds)
        r2    = r2_score(y_raw, preds)
        print(f"\nSurrogate evaluation (raw error scale):")
        print(f"  Samples  : {len(y_raw)}")
        print(f"  MAE      : {mae:.6f}")
        print(f"  R²       : {r2:.4f}")
        if r2 < 0.7:
            print("  WARNING: R² below 0.7. Consider more simulations or 'log' cleaning.")
        return {"mae": mae, "r2": r2}

    def feature_importance(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.feature_names,
                              "importance": self.model.feature_importances_})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    def save(self, path: str = "pid_surrogate.pkl"):
        joblib.dump(self, path)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path: str = "pid_surrogate.pkl") -> "PIDSurrogateModel":
        m = joblib.load(path)
        print(f"Model loaded ← {path}")
        return m


# ── PID Optimiser ─────────────────────────────────────────────────────────────

class PIDOptimiser:
    """Finds 6 PID values that minimise predicted error for given car params."""

    def __init__(self, surrogate: PIDSurrogateModel):
        self.surrogate = surrogate

    def _make_objective(self, car_params: dict):
        car_row = pd.DataFrame([car_params])
        def objective(pid_values: np.ndarray) -> float:
            pid_dict = dict(zip(PID_COLS, pid_values))
            row = pd.concat([car_row.reset_index(drop=True),
                             pd.DataFrame([pid_dict])], axis=1)
            return float(self.surrogate.predict(row)[0])
        return objective

    def _get_bounds(self, overrides: dict = None) -> list:
        bounds = []
        for col in PID_COLS:
            lo, hi = self.surrogate.pid_bounds.get(col, (0.0, 10.0))
            if overrides and col in overrides:
                lo, hi = overrides[col]
            bounds.append((lo, hi))
        return bounds

    def optimise(
        self,
        car_params: dict,
        method: str = "differential_evolution",
        pid_bounds_override: dict = None,
        n_restarts: int = 8,
        verbose: bool = True,
    ) -> dict:
        """
        Find optimal PID values for given car parameters.

        Args:
            car_params:           Dict with all 9 CAR_PARAM_COLS keys.
            method:               'differential_evolution' (global, recommended)
                                  'local' (faster, multi-start L-BFGS-B)
            pid_bounds_override:  Override search bounds per PID, e.g.
                                  {'speed_kp': (0.5, 8.0)}
            n_restarts:           Random restarts for 'local' method.
            verbose:              Print progress and result.

        Returns:
            Dict of optimal PID values + 'predicted_error'.
        """
        missing = [k for k in CAR_PARAM_COLS if k not in car_params]
        if missing:
            raise ValueError(f"Missing car parameters: {missing}")

        objective = self._make_objective(car_params)
        bounds    = self._get_bounds(pid_bounds_override)

        if method == "differential_evolution":
            if verbose:
                print("Running differential evolution (global search)...")
            res = differential_evolution(objective, bounds, seed=42,
                                         maxiter=400, tol=1e-7,
                                         workers=1, polish=True, popsize=15)
            best_pid, best_error = res.x, res.fun

        elif method == "local":
            if verbose:
                print(f"Running local optimiser ({n_restarts} restarts)...")
            rng = np.random.default_rng(42)
            best_error, best_pid = np.inf, None
            for _ in range(n_restarts):
                x0  = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
                res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 1000, "ftol": 1e-10})
                if res.fun < best_error:
                    best_error, best_pid = res.fun, res.x
        else:
            raise ValueError(f"Unknown method '{method}'.")

        result = {pid: float(v) for pid, v in zip(PID_COLS, best_pid)}
        result["predicted_error"] = float(best_error)

        if verbose:
            print("\nOptimal PID values found:")
            print(f"  {'Parameter':<35} Value")
            print(f"  {'-'*45}")
            for k, v in result.items():
                print(f"  {k:<35} {v:.6f}")

        return result


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(
    filepath: str,
    save_path: str = "pid_surrogate.pkl",
    test_size: float = 0.2,
    cleaning_method: str = "auto",
    iqr_multiplier: float = 3.0,
    percentile_cap: float = 99.0,
    abs_error_cap: float = None,
    verbose: bool = True,
) -> tuple:
    """
    Full training pipeline with automatic outlier handling.

    Args:
        filepath:         Path to your CSV data file.
        save_path:        Where to save the trained model.
        test_size:        Fraction held out for testing (default 0.2).
        cleaning_method:  'auto' (inspects data and chooses), 'iqr',
                          'percentile', 'cap', or 'log'.
        iqr_multiplier:   Controls IQR aggressiveness (default 3.0).
        percentile_cap:   Percentile threshold for 'percentile' method.
        abs_error_cap:    Hard ceiling for 'cap' method.
        verbose:          Print XGBoost training progress.

    Returns:
        (trained_surrogate, metrics_dict)
    """
    print("=" * 55)
    print("  PID Surrogate — Training Pipeline")
    print("=" * 55)

    X, y = load_data(filepath)

    X_clean, y_clean, transform_info = clean_outliers(
        X, y, method=cleaning_method,
        iqr_multiplier=iqr_multiplier,
        percentile_cap=percentile_cap,
        abs_error_cap=abs_error_cap,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=42
    )
    # For evaluation always use raw-scale y
    y_test_raw = np.expm1(y_test) if transform_info.get("log_transform") else y_test

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    surrogate = PIDSurrogateModel()
    surrogate.fit(X_train, y_train, transform_info=transform_info, verbose=verbose)

    print("\n── Test set performance ──────────────────────────")
    metrics = surrogate.evaluate(X_test, y_test_raw)

    print("\n── Top feature importances ───────────────────────")
    print(surrogate.feature_importance().head(10).to_string(index=False))

    surrogate.save(save_path)
    return surrogate, metrics


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_optimal_pid(
    car_params: dict,
    model_path: str = "pid_surrogate.pkl",
    method: str = "differential_evolution",
    pid_bounds_override: dict = None,
) -> dict:
    """
    Load saved surrogate and return optimal PIDs for a new car.
    """
    surrogate = PIDSurrogateModel.load(model_path)
    return PIDOptimiser(surrogate).optimise(
        car_params, method=method, pid_bounds_override=pid_bounds_override
    )


if __name__ == "__main__":
    import os

    CSV_PATH   = "DM_results_2.csv"
    MODEL_PATH = "pid_surrogate_2.pkl"

    df = pd.read_csv(CSV_PATH)

    # Inspect the gap — where does stable end and unstable begin?
    print(df["simulation_error"].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))

    # Hard filter: keep only the stable regime
    # Adjust the threshold based on what you see above — likely somewhere between 10–100
    STABLE_THRESHOLD = 100  # tune this
    df_stable = df[df["simulation_error"] <= STABLE_THRESHOLD]
    print(f"Kept {len(df_stable)} / {len(df)} samples ({100*len(df_stable)/len(df):.1f}%)")

    df_stable.to_csv("stable_data.csv", index=False)

    CSV_PATH = "stable_data.csv"

    x, y = load_data(CSV_PATH)

    cleaning_method = recommend_cleaning_strategy(y)
    surrogate, metrics = train(CSV_PATH, save_path=MODEL_PATH, cleaning_method=cleaning_method)

    # print("\n" + "=" * 55)
    # print("  Inference — Finding Optimal PIDs for a New Car")
    # print("=" * 55)

    # new_car = {
    #     "mass":                           1600,
    #     "max_acceleration":               6.0,
    #     "yaw_inertia":                    2200,
    #     "length":                         4.4,
    #     "max_steering_stiffness_front":   52000,
    #     "max_steering_stiffness_rear":    48000,
    #     "tyre_friction":                  0.90,
    #     "cd":                             0.32,
    #     "cross_sectional_area":           2.1,
    # }

    # result = predict_optimal_pid(new_car, model_path=MODEL_PATH, method="local")
