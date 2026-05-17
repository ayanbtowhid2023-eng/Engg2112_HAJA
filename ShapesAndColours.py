# The likes of which I've never seen!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error, r2_score

STEER_TARGET_COLS = ["opt_steer_kp", "opt_steer_ki", "opt_steer_kd"]
SPEED_TARGET_COLS = ["opt_speed_kp", "opt_speed_ki", "opt_speed_kd"]
CAR_PARAM_COLS = [
    "mass", "max_acceleration", "yaw_inertia", "length",
    "max_steering_stiffness_front", "max_steering_stiffness_rear",
    "tyre_friction", "cd", "cross_sectional_area",
]

def load_diagnostics(path: str = "training_diagnostics.json") -> dict:
    with open(path, "r") as f:
        raw = json.load(f)
    return {
        "X_test":     pd.DataFrame(raw["X_test"]),
        "ys_test":    pd.DataFrame(raw["ys_test"]),
        "yv_test":    pd.DataFrame(raw["yv_test"]),
        "steer_pred": pd.DataFrame(raw["steer_pred"]),
        "speed_pred": pd.DataFrame(raw["speed_pred"]),
    }

def ImportanceHistogram(csv1, csv2):
    # If you only want to do one of these (steer/speed) just pass None in as the other and it will skip it
    if csv1 is not None:
        df = pd.read_csv(csv1)

        features = df.iloc[:, 0]
        P = df.iloc[:, 1]
        I = df.iloc[:, 2]
        D = df.iloc[:, 3]

        x = np.arange(len(features))
        width = 0.25

        plt.figure(figsize=(12,6))

        plt.bar(x - width, P, width, label='P')
        plt.bar(x, I, width, label='I')
        plt.bar(x + width, D, width, label='D')

        plt.xticks(x, features, rotation=45, ha='right')

        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Steering PID Feature Importances")

        plt.legend()

        plt.tight_layout()

    if csv2 is None:
        plt.show()
        return
    
    df = pd.read_csv(csv2)

    features = df.iloc[:, 0]
    P = df.iloc[:, 1]
    I = df.iloc[:, 2]
    D = df.iloc[:, 3]

    x = np.arange(len(features))
    width = 0.25

    plt.figure(figsize=(12,6))

    plt.bar(x - width, P, width, label='P')
    plt.bar(x, I, width, label='I')
    plt.bar(x + width, D, width, label='D')

    plt.xticks(x, features, rotation=45, ha='right')

    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Speed PID Feature Importances")

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_feature_importance_heatmaps(
    steer_csv_path: str = "steer_feature_importance.csv",
    speed_csv_path: str = "speed_feature_importance.csv",
) -> None:
    """
    Load feature importance CSVs produced by DirectPIDModel.feature_importance()
    and render a side-by-side heatmap for steering and speed PID gains.
    """
    steer_df = pd.read_csv(steer_csv_path, index_col="feature")
    speed_df = pd.read_csv(speed_csv_path, index_col="feature")

    steer_norm = steer_df.copy()
    speed_norm = speed_df.copy()

    steer_norm.columns = ["Kp", "Ki", "Kd"]
    speed_norm.columns = ["Kp", "Ki", "Kd"]

    # Friendlier row labels
    label_map = {
        "mass":                           "Mass",
        "max_acceleration":               "Max Acceleration",
        "yaw_inertia":                    "Yaw Inertia",
        "length":                         "Length",
        "max_steering_stiffness_front":   "Stiffness Front",
        "max_steering_stiffness_rear":    "Stiffness Rear",
        "tyre_friction":                  "Tyre Friction",
        "cd":                             "Drag Coeff (Cd)",
        "cross_sectional_area":           "Cross-section Area",
        "aero_drag_proxy":                "Aero Drag Proxy",
        "steer_balance":                  "Steer Balance",
        "lateral_grip_front":             "Lateral Grip Front",
        "lateral_grip_rear":              "Lateral Grip Rear",
        "accel_per_mass":                 "Accel / Mass",
        "yaw_factor":                     "Yaw Factor",
        "stiffness_per_mass":             "Stiffness / Mass",
        "grip_force":                     "Grip Force",
        "drag_to_mass":                   "Drag / Mass",
    }
    steer_norm.index = [label_map.get(f, f) for f in steer_norm.index]
    speed_norm.index = [label_map.get(f, f) for f in speed_norm.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    for ax, df, title in [
        (ax1, steer_norm, "Steering PID — Feature Importance"),
        (ax2, speed_norm, "Speed PID — Feature Importance"),
    ]:
        im = ax.imshow(df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=df.values.max())

        # Axis ticks
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=9)

        # Annotate cells with the raw (un-normalised) value
        raw_df = steer_df if ax is ax1 else speed_df
        raw_df.columns = ["Kp", "Ki", "Kd"]
        raw_df.index = df.index
        for row in range(len(df.index)):
            for col in range(len(df.columns)):
                val = raw_df.iloc[row, col]
                text_color = "white" if df.iloc[row, col] > 0.6 else "black"
                ax.text(col, row, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color=text_color)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        fig.colorbar(im, ax=ax, label="Feature Importance", shrink=0.8)

    plt.suptitle("Feature Importance by PID Gain", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()



def plot_predicted_vs_actual(diag_path: str = "training_diagnostics.json") -> None:
    d = load_diagnostics(diag_path)
    steer_actual, steer_pred = d["ys_test"], d["steer_pred"]
    speed_actual, speed_pred = d["yv_test"], d["speed_pred"]

    titles = ["Steer Kp", "Steer Ki", "Steer Kd",
              "Speed Kp", "Speed Ki", "Speed Kd"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (ax, title) in enumerate(zip(axes, titles)):
        if i < 3:
            col = STEER_TARGET_COLS[i]
            y_true = steer_actual[col].values
            y_pred = steer_pred[col].values
        else:
            col = SPEED_TARGET_COLS[i - 3]
            y_true = speed_actual[col].values
            y_pred = speed_pred[col].values

        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.5, s=18, color="steelblue", edgecolors="none")

        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{title}\nMAE={mae:.4f}  R²={r2:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Predicted vs Actual PID Gains", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()



def plot_error_distributions(diag_path: str = "training_diagnostics.json") -> None:
    d = load_diagnostics(diag_path)
    steer_actual, steer_pred = d["ys_test"], d["steer_pred"]
    speed_actual, speed_pred = d["yv_test"], d["speed_pred"]

    titles = ["Steer Kp", "Steer Ki", "Steer Kd",
              "Speed Kp", "Speed Ki", "Speed Kd"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (ax, title) in enumerate(zip(axes, titles)):
        if i < 3:
            col = STEER_TARGET_COLS[i]
            residuals = steer_pred[col].values - steer_actual[col].values
        else:
            col = SPEED_TARGET_COLS[i - 3]
            residuals = speed_pred[col].values - speed_actual[col].values

        ax.hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
        ax.axvline(residuals.mean(), color="orange", linestyle="-", linewidth=1.2,
                   label=f"Mean={residuals.mean():.4f}")

        ax.set_xlabel("Residual (Predicted − Actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nStd={residuals.std():.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Prediction Error Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()



def plot_error_vs_car_params(diag_path: str = "training_diagnostics.json") -> None:
    d = load_diagnostics(diag_path)
    X_test = d["X_test"]
    steer_actual, steer_pred = d["ys_test"], d["steer_pred"]
    speed_actual, speed_pred = d["yv_test"], d["speed_pred"]

    param_labels = {
        "mass":                         "Mass [kg]",
        "max_acceleration":             "Max Accel [m/s²]",
        "yaw_inertia":                  "Yaw Inertia [kg·m²]",
        "length":                       "Length [m]",
        "max_steering_stiffness_front": "Stiffness Front",
        "max_steering_stiffness_rear":  "Stiffness Rear",
        "tyre_friction":                "Tyre Friction (μ)",
        "cd":                           "Drag Coeff (Cd)",
        "cross_sectional_area":         "Cross-section [m²]",
    }

    gain_configs = [
        ("Steer Kp", STEER_TARGET_COLS[0], steer_actual, steer_pred),
        ("Steer Ki", STEER_TARGET_COLS[1], steer_actual, steer_pred),
        ("Steer Kd", STEER_TARGET_COLS[2], steer_actual, steer_pred),
        ("Speed Kp", SPEED_TARGET_COLS[0], speed_actual, speed_pred),
        ("Speed Ki", SPEED_TARGET_COLS[1], speed_actual, speed_pred),
        ("Speed Kd", SPEED_TARGET_COLS[2], speed_actual, speed_pred),
    ]

    n_params = len(CAR_PARAM_COLS)
    cols = 3
    rows = int(np.ceil(n_params / cols))

    for gain_title, col, actual_df, pred_df in gain_configs:
        abs_error = np.abs(pred_df[col].values - actual_df[col].values)

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2))
        axes = axes.flatten()

        for j, param in enumerate(CAR_PARAM_COLS):
            ax = axes[j]
            ax.scatter(X_test[param].values, abs_error,
                       alpha=0.5, s=14, color="steelblue", edgecolors="none")

            z = np.polyfit(X_test[param].values, abs_error, 1)
            x_line = np.linspace(X_test[param].min(), X_test[param].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.2)

            ax.set_xlabel(param_labels.get(param, param), fontsize=8)
            ax.set_ylabel("Abs Error", fontsize=8)
            ax.grid(True, alpha=0.3)

        for k in range(n_params, len(axes)):
            axes[k].set_visible(False)

        fig.suptitle(f"Abs Error vs Car Parameters — {gain_title}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

    plt.show()

def main():
    ImportanceHistogram("steer_feature_importance.csv", "speed_feature_importance.csv")
    plot_feature_importance_heatmaps(
        steer_csv_path="steer_feature_importance.csv",
        speed_csv_path="speed_feature_importance.csv",
    )
    plot_predicted_vs_actual()
    plot_error_distributions()
    plot_error_vs_car_params()

if __name__ == "__main__":
    main()
