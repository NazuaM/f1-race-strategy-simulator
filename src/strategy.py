import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import linregress

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import train_dataset, train_dataset_sorted


COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
FALLBACK_DEGRADATION = {"HARD": 0.03, "MEDIUM": 0.04, "SOFT": 0.06}
RACE_NAME_ALIASES = {
    "Austin": {"Austin", "United States", "United States Grand Prix"},
}


def get_circuit_pit_loss(year, gp_name, driver, race_dataset=None):
    path = hf_hub_download(
        repo_id="VforVitorio/f1-strategy-dataset",
        filename=f"data/raw/{year}/{gp_name}/pitstops.parquet",
        repo_type="dataset",
    )
    pitstops = pd.read_parquet(path)
    driver_pits = pitstops[pitstops["Driver"] == driver].copy()
    driver_pits["LapTime_s"] = pd.to_timedelta(
        driver_pits["LapTime"]
    ).dt.total_seconds()
    if race_dataset is not None:
        race_names = RACE_NAME_ALIASES.get(gp_name, {gp_name})
        race_laps = race_dataset[
            (race_dataset["Year"] == year)
            & race_dataset["GP_Name"].isin(race_names)
            & (race_dataset["Driver"] == driver)
        ]
        median_lap = race_laps["LapTime_s"].median()
    else:
        clean_laps = driver_pits[
            driver_pits["PitInTime"].isna() & driver_pits["PitOutTime"].isna()
        ]
        median_lap = clean_laps["LapTime_s"].median()
    if pd.isna(median_lap):
        raise ValueError(f"No clean laps found for {driver} at {gp_name} {year}")

    in_lap = driver_pits[driver_pits["PitInTime"].notna()].sort_values("LapNumber")
    if in_lap.empty:
        raise ValueError(f"No pit-in lap found for {driver} at {gp_name} {year}")
    pit_lap = in_lap.iloc[0]["LapNumber"]
    out_lap = driver_pits[
        (driver_pits["PitOutTime"].notna())
        & (driver_pits["LapNumber"] > pit_lap)
    ].sort_values("LapNumber")
    if out_lap.empty and race_dataset is not None:
        out_lap = race_laps[race_laps["LapNumber"] == pit_lap + 1]
    if out_lap.empty:
        print(
            f"Warning: no out-lap record found after lap {pit_lap}; "
            "using the clean median as the out-lap estimate."
        )
        out_lap_time = median_lap
    else:
        out_lap_time = (
            out_lap.iloc[0]["LapTime_s"]
            if "LapTime_s" in out_lap
            else pd.to_timedelta(out_lap.iloc[0]["LapTime"]).total_seconds()
        )

    in_lap_time = in_lap.iloc[0]["LapTime_s"]

    print("Pit-in lap time:", in_lap_time)
    print("Out-lap time:", out_lap_time)
    print("Median clean lap:", median_lap)

    return float(
        in_lap_time + out_lap_time
        - 2 * median_lap
    )


def build_stint_table(dataset):
    rows = []
    for keys, group in dataset.groupby(["Year", "GP_Name", "Driver", "Stint"]):
        stint = group.sort_values("LapNumber").dropna(
            subset=["TyreLife", "FuelAdjustedLapTime", "DegradationRate"]
        )
        if len(stint) < 5 or stint["TyreLife"].nunique() < 2:
            continue

        trend = linregress(stint["TyreLife"], stint["FuelAdjustedLapTime"])
        rows.append(
            {
                "Year": keys[0],
                "GP_Name": keys[1],
                "Driver": keys[2],
                "Stint": keys[3],
                "Compound": stint["Compound"].mode().iloc[0],
                "Laps": len(stint),
                "SecondsPerTyreLap": trend.slope,
                "ExistingRateMean": stint["DegradationRate"].mean(),
                "R2": trend.rvalue**2,
            }
        )
    return pd.DataFrame(rows)


def build_race_summary(pace_table):
    reliable_pace = pace_table[
        pace_table["Compound"].isin(COMPOUNDS)
        & (pace_table["Laps"] >= 8)
        & (pace_table["R2"] >= 0.3)
    ]
    race_summary = (
        reliable_pace.groupby(["Year", "GP_Name", "Compound"])
        .agg(
            Stints=("SecondsPerTyreLap", "count"),
            MedianDeg=("SecondsPerTyreLap", "median"),
        )
        .reset_index()
    )
    race_summary["UseRaceData"] = (
        (race_summary["Stints"] >= 5)
        & race_summary["MedianDeg"].between(0, 0.25)
    )
    overall = reliable_pace.groupby("Compound")["SecondsPerTyreLap"].median()
    race_summary["StrategyData"] = race_summary.apply(
        lambda row: (
            row["MedianDeg"]
            if row["UseRaceData"]
            else overall.get(row["Compound"], FALLBACK_DEGRADATION[row["Compound"]])
        ),
        axis=1,
    )
    return race_summary, overall


def calculate_historical_pace_offsets(
    dataset,
    gp_name,
    max_tyre_life=5,
    minimum_rows=20,
):
    training_dataset_offset = dataset[
        (dataset["GP_Name"] == gp_name)
        & (dataset["Compound"].isin(["HARD", "MEDIUM", "SOFT"]))
        & (dataset["TyreLife"] <= max_tyre_life)
    ]

    training_dataset_summary = (
        training_dataset_offset
        .groupby("Compound")
        .agg(
            Rows=("Compound", "size"),
            MedianLapTime=("FuelAdjustedLapTime", "median"),
        )
    )

    if "MEDIUM" not in training_dataset_summary.index:
      raise ValueError(f"No Medium reference data found for {gp_name}")

    medium_time = training_dataset_summary.loc["MEDIUM", "MedianLapTime"]

    training_dataset_summary["OffsetFromMedium"] = (
        training_dataset_summary["MedianLapTime"] - medium_time
    )

    training_dataset_summary["Reliable"] = (
        training_dataset_summary["Rows"] >= minimum_rows
    )

    return training_dataset_summary


def current_pace_anchor(
    dataset,
    year,
    gp_name,
    driver,
    decision_lap,
    window=5,
):
    """Return the driver's recent pace and mean tyre age before a decision."""
    laps = dataset[
        (dataset["Year"] == year)
        & (dataset["GP_Name"] == gp_name)
        & (dataset["Driver"] == driver)
        & (dataset["LapNumber"] <= decision_lap)
    ].sort_values("LapNumber")
    if laps.empty:
        raise ValueError(f"No laps found for {driver} at {gp_name} {year}")

    current_stint = laps["Stint"].iloc[-1]
    laps = laps[laps["Stint"] == current_stint].tail(window)
    if laps.empty:
        raise ValueError("No current-stint laps available for pace anchor")
    return (
        float(laps["FuelAdjustedLapTime"].median()),
        float(laps["TyreLife"].mean()),
    )


def get_degradation_rate(
    year,
    gp_name,
    compound,
    race_summary,
    overall_compound,
):
    """Resolve degradation using exact race, historical race, then global data."""
    exact = race_summary[
        (race_summary["Year"] == year)
        & (race_summary["GP_Name"] == gp_name)
        & (race_summary["Compound"] == compound)
        & race_summary["UseRaceData"]
    ]
    if not exact.empty:
        return float(exact["StrategyData"].iloc[0])

    historical = race_summary[
        (race_summary["GP_Name"] == gp_name)
        & (race_summary["Compound"] == compound)
        & race_summary["UseRaceData"]
    ]
    if not historical.empty:
        return float(historical["StrategyData"].median())

    if compound in overall_compound:
        return float(overall_compound[compound])
    if compound in FALLBACK_DEGRADATION:
        return FALLBACK_DEGRADATION[compound]
    raise ValueError(f"No degradation rate available for {compound}")


def calculate_stint_life_limits(dataset, gp_name, quantile=0.95):
    """Estimate circuit-specific tyre-life limits from historical stints."""
    circuit = dataset[
        (dataset["GP_Name"] == gp_name)
        & dataset["Compound"].isin(COMPOUNDS)
    ]
    if circuit.empty:
        circuit = dataset[dataset["Compound"].isin(COMPOUNDS)]
    stint_lengths = (
        circuit.groupby(["Year", "GP_Name", "Driver", "Stint", "Compound"])
        .size()
        .rename("Laps")
        .reset_index()
    )
    limits = (
        stint_lengths.groupby("Compound")["Laps"]
        .quantile(quantile)
        .apply(np.ceil)
        .astype(int)
        .to_dict()
    )
    return limits


def assess_strategy_feasibility(
    current_compound,
    current_age,
    laps_remaining,
    available_compounds,
    stint_life_limits,
):
    required_laps = {
        "STAY_OUT": current_age + laps_remaining,
        **{
            f"PIT_FOR_{compound}": laps_remaining
            for compound in available_compounds
            if compound != current_compound
        },
    }
    feasibility = {"STAY_OUT": required_laps["STAY_OUT"] <= stint_life_limits.get(
        current_compound, 0
    )}
    for compound in available_compounds:
        if compound == current_compound:
            continue
        feasibility[f"PIT_FOR_{compound}"] = (
            required_laps[f"PIT_FOR_{compound}"]
            <= stint_life_limits.get(compound, 0)
        )
    return feasibility, required_laps


def project_strategy_totals(
    pace_now,
    recent_age,
    current_compound,
    current_age,
    laps_remaining,
    pit_loss,
    degradation,
    pace_offsets,
    available_compounds,
    stint_life_limits=None,
):
    missing = set(available_compounds) - set(pace_offsets)
    if missing or current_compound not in pace_offsets:
        raise ValueError(
            "Missing reliable pace offsets for: "
            + ", ".join(sorted(missing | {current_compound} - set(pace_offsets)))
        )

    fresh_current = pace_now - degradation[current_compound] * recent_age
    fresh_medium = fresh_current - pace_offsets[current_compound]
    laps = np.arange(laps_remaining)
    totals = {
        "STAY_OUT": float(
            (
                fresh_current
                + degradation[current_compound] * (current_age + laps)
            ).sum()
        )
    }
    for compound in available_compounds:
        if compound == current_compound:
            continue
        fresh_target = fresh_medium + pace_offsets[compound]
        totals[f"PIT_FOR_{compound}"] = float(
            pit_loss
            + (
                fresh_target
                + degradation[compound] * laps
            ).sum()
        )
    return totals


def evaluate_all_strategies(
    year,
    gp_name,
    current_compound,
    current_age,
    laps_remaining,
    baseline_pace,
    pit_loss,
    dataset,
    race_summary,
    overall_compound,
    pace_offsets,
    available_compounds,
    recent_age=None,
    stint_life_limits=None,
):
    if stint_life_limits is None:
        stint_life_limits = {}
    degradation = {
        compound: get_degradation_rate(
            year, gp_name, compound, race_summary, overall_compound
        )
        for compound in available_compounds
    }
    if current_compound not in degradation:
        degradation[current_compound] = get_degradation_rate(
            year, gp_name, current_compound, race_summary, overall_compound
        )
    totals = project_strategy_totals(
        baseline_pace,
        current_age if recent_age is None else recent_age,
        current_compound,
        current_age,
        laps_remaining,
        pit_loss,
        degradation,
        pace_offsets,
        available_compounds,
        stint_life_limits,
    )
    feasibility, required_laps = assess_strategy_feasibility(
        current_compound,
        current_age,
        laps_remaining,
        available_compounds,
        stint_life_limits,
    ) if stint_life_limits else (
        {strategy: True for strategy in totals},
        {},
    )

    result = pd.DataFrame(
        [
            {
                "Strategy": strategy,
                "Projected Total Time (s)": (
                    totals[strategy] if feasibility[strategy] else np.nan
                ),
                "Feasible": feasibility[strategy],
                "Required tyre laps": required_laps.get(strategy),
                "Estimated tyre limit": (
                    stint_life_limits.get(current_compound)
                    if strategy == "STAY_OUT"
                    else stint_life_limits.get(
                        strategy.removeprefix("PIT_FOR_")
                    )
                ),
            }
            for strategy in totals
        ]
    )
    result["Projected Total Time (s)"] = result[
        "Projected Total Time (s)"
    ].round(2)
    feasible_totals = [
        totals[strategy] for strategy, allowed in feasibility.items() if allowed
    ]
    reference_total = min(feasible_totals) if feasible_totals else np.nan
    result["Delta to Stay Out (s)"] = (
        result["Projected Total Time (s)"] - totals["STAY_OUT"]
    ).round(2)
    result["Status"] = np.where(result["Feasible"], "Feasible", "Exceeds tyre-life limit")
    return result.sort_values(
        ["Feasible", "Projected Total Time (s)"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def run_monte_carlo_strategy(
    year,
    gp_name,
    current_compound,
    current_age,
    laps_remaining,
    baseline_pace,
    pit_loss,
    simulations,
    dataset,
    race_summary,
    overall_compound,
    pace_offsets,
    available_compounds,
    recent_age=None,
    seed=42,
    stint_life_limits=None,
):
    rng = np.random.default_rng(seed)
    strategies = ["STAY_OUT"] + [
        f"PIT_FOR_{compound}"
        for compound in available_compounds
        if compound != current_compound
    ]
    wins = dict.fromkeys(strategies, 0)
    if stint_life_limits:
        feasibility, _ = assess_strategy_feasibility(
            current_compound,
            current_age,
            laps_remaining,
            available_compounds,
            stint_life_limits,
        )
        strategies = [strategy for strategy in strategies if feasibility[strategy]]
        if not strategies:
            return {"NO_FEASIBLE_STRATEGY": 100.0}
        wins = dict.fromkeys(strategies, 0)
    base_degradation = {
        compound: get_degradation_rate(
            year, gp_name, compound, race_summary, overall_compound
        )
        for compound in available_compounds
    }
    base_degradation[current_compound] = get_degradation_rate(
        year, gp_name, current_compound, race_summary, overall_compound
    )
    for _ in range(simulations):
        degradation = {
            compound: max(
                0.0,
                rng.normal(rate, max(rate * 0.25, 0.005)),
            )
            for compound, rate in base_degradation.items()
        }
        offsets = {
            compound: rng.normal(offset, 0.15)
            for compound, offset in pace_offsets.items()
        }
        totals = project_strategy_totals(
            baseline_pace,
            current_age if recent_age is None else recent_age,
            current_compound,
            current_age,
            laps_remaining,
            max(0.0, rng.normal(pit_loss, 1.5)),
            degradation,
            offsets,
            available_compounds,
            stint_life_limits,
        )
        feasible_totals = {
            strategy: total
            for strategy, total in totals.items()
            if strategy in wins
        }
        if feasible_totals:
            wins[min(feasible_totals, key=feasible_totals.get)] += 1
    return {strategy: count / simulations * 100 for strategy, count in wins.items()}


def main():
    pace_table = build_stint_table(train_dataset_sorted)
    race_summary, overall_compound = build_race_summary(pace_table)
    Path("models").mkdir(exist_ok=True)
    race_summary.to_csv("models/race_sumarry.csv", index=False)
    race_names = RACE_NAME_ALIASES["Austin"]
    ver_austin = train_dataset[
        (train_dataset["Year"] == 2023)
        & train_dataset["GP_Name"].isin(race_names)
        & (train_dataset["Driver"] == "VER")
        & (train_dataset["Compound"] == "MEDIUM")
        & (train_dataset["TyreLife"] == 1)
    ]
    if ver_austin.empty:
        ver_austin = train_dataset[
            (train_dataset["Year"] == 2023)
            & train_dataset["GP_Name"].isin(race_names)
            & (train_dataset["Driver"] == "VER")
            & (train_dataset["TyreLife"] == 1)
        ]
    if ver_austin.empty:
        ver_austin = train_dataset[
            (train_dataset["Year"] == 2023)
            & train_dataset["GP_Name"].isin(race_names)
            & (train_dataset["Driver"] == "VER")
        ]
    if ver_austin.empty:
        ver_austin = train_dataset[
            (train_dataset["Year"] == 2023)
            & (train_dataset["Driver"] == "VER")
        ]
    if ver_austin.empty:
        raise ValueError("No 2023 VER laps found for baseline pace")
    if not set(ver_austin["GP_Name"]).issubset(race_names):
        print("Warning: Austin 2023 VER data was unavailable; using all 2023 VER laps.")
    baseline_pace, recent_age = current_pace_anchor(
        train_dataset, 2023, "Austin", "VER", int(ver_austin["LapNumber"].max())
    )
    pit_loss = 22.0
    offsets_table = calculate_historical_pace_offsets(train_dataset, "Austin")
    pace_offsets = offsets_table[
        offsets_table["Reliable"]
    ]["OffsetFromMedium"].to_dict()
    available_compounds = list(pace_offsets)

    print(
        evaluate_all_strategies(
            2023,
            "Austin",
            "MEDIUM",
            7,
            15,
            baseline_pace,
            pit_loss,
            train_dataset,
            race_summary,
            overall_compound,
            pace_offsets,
            available_compounds,
            recent_age,
        )
    )
    print(
        run_monte_carlo_strategy(
            2023,
            "Austin",
            "MEDIUM",
            20,
            18,
            baseline_pace,
            pit_loss,
            1000,
            train_dataset,
            race_summary,
            overall_compound,
            pace_offsets,
            available_compounds,
        )
    )


if __name__ == "__main__":
    main()
