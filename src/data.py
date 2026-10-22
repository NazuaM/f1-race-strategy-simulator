import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


REPO_URL = (
    "https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset"
    "/resolve/main/data/processed/laps_featured_{year}.parquet"
)
GROUP_COLUMNS = ["Year", "GP_Name", "Driver", "Stint"]
REQUIRED_COLUMNS = [
    "Prev_LapTime",
    "LapTime_s",
    "Prev_DegradationRate",
    "Prev_CumulativeDeg",
    "Prev_DegAcceleration",
]
FEATURES = [
    "DriverNumber",
    "LapNumber",
    "Stint",
    "TyreLife",
    "FreshTyre",
    "Position",
    "CompoundID",
    "TeamID",
    "LapsSincePitStop",
    "FuelLoad",
    "FuelEffect",
    "Prev_LapTime",
    "Prev_TyreLife",
    "Prev_SpeedST",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Rainfall",
    "laps_remaining",
    "Cluster",
    "mean_sector_speed",
    "Prev_DegradationRate",
    "Prev_CumulativeDeg",
    "Prev_DegAcceleration",
]


def load_years(years: list[int]) -> pd.DataFrame:
    """Load and combine the processed lap data for the requested years."""
    try:
        datasets = [
            pd.read_parquet(REPO_URL.format(year=year))
            for year in years
        ]
    except Exception as error:
        raise RuntimeError(f"Unable to load lap datasets for {years}: {error}") from error

    return pd.concat(datasets, ignore_index=True)


def prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Keep consecutive laps and add lagged degradation features."""
    prepared = dataset.sort_values(GROUP_COLUMNS + ["LapNumber"]).copy()
    grouped = prepared.groupby(GROUP_COLUMNS)

    for column in ["DegradationRate", "CumulativeDeg", "DegAcceleration"]:
        prepared[f"Prev_{column}"] = grouped[column].shift(1)

    prepared["LapNumberDiff"] = grouped["LapNumber"].diff()
    prepared = prepared[prepared["LapNumberDiff"] == 1]
    return prepared.dropna(subset=REQUIRED_COLUMNS)


def evaluate_metrics(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    return mae, rmse


def print_baseline(dataset: pd.DataFrame, label: str) -> None:
    mae, rmse = evaluate_metrics(dataset["LapTime_s"], dataset["Prev_LapTime"])
    print(f"Baseline MAE (seconds) for {label}: {mae:.3f}")
    print(f"Baseline RMSE (seconds) for {label}: {rmse:.3f}")


train_dataset = load_years([2023, 2024])
test_dataset = load_years([2025])

print(f"Training dataset rows (2023-2024): {len(train_dataset):,}")
print(f"Testing dataset rows (2025): {len(test_dataset):,}")
print("Training dataset columns:", train_dataset.columns.to_list())
print("Training dataset shape:", train_dataset.shape)

train_dataset_sorted = prepare_dataset(train_dataset)
test_dataset_sorted = prepare_dataset(test_dataset)

print_baseline(train_dataset_sorted, "training dataset")
print_baseline(test_dataset_sorted, "test dataset")
print("Clean Feature Count:", len(FEATURES))
