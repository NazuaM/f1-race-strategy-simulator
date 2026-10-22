import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import linregress
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FEATURES, evaluate_metrics, test_dataset_sorted, train_dataset_sorted


def create_model():
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
    )


def get_targets(dataset):
    lap_times = dataset["LapTime_s"]
    return {
        "absolute": lap_times,
        "delta": lap_times - dataset["Prev_LapTime"],
    }


def evaluate_dataset(dataset, models=None):
    features = dataset[FEATURES]
    targets = get_targets(dataset)
    is_training = models is None
    models = models or {name: create_model() for name in targets}
    metrics = {}

    for name, target in targets.items():
        model = models[name]
        if is_training:
            model.fit(features, target)
        metrics[name] = evaluate_metrics(target, model.predict(features))

    return metrics, models


def plot_stint(stint, predicted, driver, circuit, year, stint_number):
    lap_numbers = stint["LapNumber"]
    actual = stint["LapTime_s"]

    plt.figure(figsize=(11, 5))
    plt.plot(lap_numbers, actual, marker="o", label="Actual lap time")
    plt.plot(lap_numbers, predicted, marker="o", label="Predicted lap time")
    plt.xlabel("Lap number")
    plt.ylabel("Lap time (seconds)")
    plt.title(f"{driver} - {circuit} {year}, Stint {stint_number}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.scatter(stint["TyreLife"], actual)
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Actual lap time (seconds)")
    plt.title(f"{driver} - {circuit} {year}: Pace vs Tyre Age")
    plt.grid(True, alpha=0.3)
    plt.show()

Path("models").mkdir(exist_ok=True)


train_metrics, models = evaluate_dataset(train_dataset_sorted)

models["absolute"].save_model("models/laptime_xgb.json")
test_metrics, _ = evaluate_dataset(test_dataset_sorted, models)

print("Feature Importance:")
importances = pd.DataFrame(
    {"Features": FEATURES, "Importance": models["delta"].feature_importances_}
).sort_values(by="Importance", ascending=False)
print("--- Feature Importance Breakdown ---")
print(importances.head(5))

stint_ver = test_dataset_sorted[
    (test_dataset_sorted["GP_Name"] == "Austin")
    & (test_dataset_sorted["Driver"] == "VER")
    & (test_dataset_sorted["Stint"] == 1)
].sort_values("LapNumber")

stint_features = stint_ver[FEATURES]
stint_predicted = models["absolute"].predict(stint_features)
plot_stint(stint_ver, stint_predicted, "Verstappen", "Austin", 2025, 1)


valid = stint_ver.dropna(subset=['TyreLife', 'FuelAdjustedLapTime'])

result = linregress(valid['TyreLife'], valid['FuelAdjustedLapTime'])

print(f"Estimated pace loss: {result.slope:.3f} seconds per tyre lap")
print(f"R²: {result.rvalue**2:.3f}")
print(f"Correlation: {result.rvalue:.3f}")
print(f"Estimated change over 10 tyre laps: {result.slope * 10:.3f} seconds")

# "Dataset's DegradationRate column and measured linear slope disagree (0.10 vs 0.06 sec/lap). Need to decide which to trust before using it as a feature."

print("Mean DegradationRate:",
      round(stint_ver['DegradationRate'].mean(), 4))

print("Median DegradationRate:",
      round(stint_ver['DegradationRate'].median(), 4))

print("Regression slope:",
      round(result.slope, 4))
