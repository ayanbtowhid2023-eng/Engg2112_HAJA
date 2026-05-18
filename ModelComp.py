"""
Model Comparison: XGBoost vs Linear Regression, Random Forest,
                  Gradient Boosting, and MLP Regressor
Targets: opt_steer_kp, opt_steer_ki, opt_steer_kd
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("MasterDocCSV.csv")

TARGET_COLS  = ["opt_steer_kp", "opt_steer_ki", "opt_steer_kd"]
DROP_COLS    = ["opt_speed_kp", "opt_speed_ki", "opt_speed_kd",
                "best_speed_error", "best_heading_error"] + TARGET_COLS

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS + TARGET_COLS]

X = df[FEATURE_COLS]

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}  →  {FEATURE_COLS}")
print(f"  Targets  : {TARGET_COLS}")

# ── 2. Define Models ───────────────────────────────────────────────────────────
# MLPRegressor and LinearRegression benefit from scaling, so wrap in a Pipeline
models = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    ),
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    ),
    "MLP Regressor": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        )),
    ]),
}

# ── 3. Cross-Validation ────────────────────────────────────────────────────────
CV_FOLDS = 5
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

METRICS = {
    "R²":   "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE":  "neg_mean_absolute_error",
}

# results[target][model][metric] = (mean, std)
results = {t: {} for t in TARGET_COLS}

print(f"\nRunning {CV_FOLDS}-fold cross-validation …\n")

for target in TARGET_COLS:
    y = df[target]
    print(f"  Target: {target}")
    for model_name, model in models.items():
        row = {}
        for metric_label, scoring in METRICS.items():
            scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
            if scoring.startswith("neg_"):
                scores = -scores          # convert back to positive
            row[metric_label] = (scores.mean(), scores.std())
        results[target][model_name] = row
        print(f"    ✓ {model_name}")
    print()

# ── 4. Print Results Table ─────────────────────────────────────────────────────
SEP = "=" * 70

for target in TARGET_COLS:
    print(SEP)
    print(f"TARGET: {target}")
    print(SEP)

    # Header
    header = f"{'Model':<22}"
    for m in METRICS:
        header += f"  {m:>16} {'±':>3}"
    print(header)
    print("-" * 70)

    for model_name in models:
        row_str = f"{model_name:<22}"
        for metric_label in METRICS:
            mean, std = results[target][model_name][metric_label]
            row_str += f"  {mean:>10.4f} ±{std:>6.4f}"
        print(row_str)

    # Best model per metric
    print("-" * 70)
    for metric_label in METRICS:
        if metric_label == "R²":
            best = max(models, key=lambda m: results[target][m][metric_label][0])
        else:
            best = min(models, key=lambda m: results[target][m][metric_label][0])
        best_val = results[target][best][metric_label][0]
        print(f"  Best {metric_label:<6}: {best}  ({best_val:.4f})")
    print()

# ── 5. Summary: XGBoost rank across all targets & metrics ─────────────────────
print(SEP)
print("XGBOOST RANK SUMMARY  (1 = best)")
print(SEP)

model_names = list(models.keys())

for target in TARGET_COLS:
    print(f"\n  {target}")
    for metric_label in METRICS:
        vals = [(m, results[target][m][metric_label][0]) for m in model_names]
        reverse = (metric_label == "R²")          # higher is better for R²
        ranked  = sorted(vals, key=lambda x: x[1], reverse=reverse)
        xgb_rank = next(i+1 for i, (m, _) in enumerate(ranked) if m == "XGBoost")
        print(f"    {metric_label:<6}: rank {xgb_rank}/{len(models)}  "
              f"(value = {results[target]['XGBoost'][metric_label][0]:.4f})")

print(f"\n{'=' * 70}")

# ── 6. Averaged R² and MAE across all three targets ───────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("AVERAGED METRICS ACROSS ALL TARGETS")
print(SEP)
print(f"{'Model':<22}  {'Avg R²':>10}  {'Avg MAE':>10}")
print("-" * 46)

avg_r2  = {}
avg_mae = {}

for model_name in models:
    r2_vals  = [results[t][model_name]["R²"][0]  for t in TARGET_COLS]
    mae_vals = [results[t][model_name]["MAE"][0] for t in TARGET_COLS]
    avg_r2[model_name]  = np.mean(r2_vals)
    avg_mae[model_name] = np.mean(mae_vals)
    print(f"{model_name:<22}  {avg_r2[model_name]:>10.4f}  {avg_mae[model_name]:>10.4f}")

# ── 7. Averaged Plot Only ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt

COLORS = {
    "XGBoost":           "#E6521E",
    "Linear Regression": "#4C8BF5",
    "Random Forest":     "#34A853",
    "Gradient Boosting": "#FBBC05",
    "MLP Regressor":     "#9B59B6",
}

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_title(
    "Average Model Performance\n(Steering PID Gains — 5-Fold CV)",
    fontsize=14,
    fontweight="bold",
    pad=12
)

ax.set_xlabel("Average R²", fontsize=11)
ax.set_ylabel("Average MAE", fontsize=11)

ax.grid(True, linestyle="--", alpha=0.4)

# Plot points
for model_name in models:
    ax.scatter(
        avg_r2[model_name],
        avg_mae[model_name],
        color=COLORS[model_name],
        marker="o",
        s=80,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

# Legend
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=COLORS[m],
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=8,
        linestyle="",
        label=m,
    )
    for m in models
]

ax.legend(
    handles=handles,
    fontsize=10,
    frameon=True,
    title="Models",
    title_fontsize=10,
    loc="best",
)

plt.tight_layout()
plt.show()
