import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import test_dataset, train_dataset, train_dataset_sorted
from src.strategy import (
    build_race_summary,
    build_stint_table,
    calculate_historical_pace_offsets,
    current_pace_anchor,
    calculate_stint_life_limits,
    evaluate_all_strategies,
    get_degradation_rate,
    run_monte_carlo_strategy,
)


pace_table = build_stint_table(train_dataset_sorted)
race_summary, overall_compound = build_race_summary(pace_table)

baseline_pace, recent_age = current_pace_anchor(
    test_dataset, 2025, "Monaco", "VER", 27
)
pit_loss = 22.0

monaco_offsets = calculate_historical_pace_offsets(train_dataset, "Monaco")
reliable_offsets = (
    monaco_offsets[monaco_offsets["Reliable"]]["OffsetFromMedium"].to_dict()
)
available_compounds = list(reliable_offsets)
stint_life_limits = calculate_stint_life_limits(train_dataset, "Monaco")

result = evaluate_all_strategies(
    year=2025,
    gp_name="Monaco",
    current_compound="HARD",
    current_age=27,
    laps_remaining=49,
    baseline_pace=baseline_pace,
    pit_loss=pit_loss,
    dataset=test_dataset,
    race_summary=race_summary,
    overall_compound=overall_compound,
    pace_offsets=reliable_offsets,
    available_compounds=available_compounds,
    recent_age=recent_age,
    stint_life_limits=stint_life_limits,
)

monte_carlo_probabilities = run_monte_carlo_strategy(
    year=2025,
    gp_name="Monaco",
    current_compound="HARD",
    current_age=27,
    laps_remaining=49,
    baseline_pace=baseline_pace,
    pit_loss=pit_loss,
    simulations=1000,
    dataset=test_dataset,
    race_summary=race_summary,
    overall_compound=overall_compound,
    pace_offsets=reliable_offsets,
    available_compounds=available_compounds,
    recent_age=recent_age,
    stint_life_limits=stint_life_limits,
)

recommended_action = result.iloc[0]["Strategy"]
actual_action = "PIT_FOR_MEDIUM"

facts = {
    "race": {
        "year": 2025,
        "circuit": "Monaco",
        "driver": "VER",
        "decision_lap": 27,
        "current_compound": "HARD",
        "current_tyre_age": 27,
        "laps_remaining": 49,
        "baseline_pace": baseline_pace,
        "recent_age": recent_age,
        "pit_loss": pit_loss,
        "stint_life_limits": stint_life_limits,
    },
    "tyre_rates": {
        compound: get_degradation_rate(
            2025, "Monaco", compound, race_summary, overall_compound
        )
        for compound in available_compounds
    },
    "strategy_table": result.to_dict(orient="records"),
    "monte_carlo_probabilities": monte_carlo_probabilities,
}

with open("models/monaco_ver_strategy_facts.json", "w") as file:
    json.dump(facts, file, indent=2)

evaluation_case = pd.DataFrame(
    [
        {
            "Year": 2025,
            "GP_Name": "Monaco",
            "Driver": "VER",
            "DecisionLap": 27,
            "CurrentCompound": "HARD",
            "CurrentTyreAge": 27,
            "ActualAction": actual_action,
            "laps_remaining": 49,
            "SimulatorAction": recommended_action,
            "SimulatorAgrees": recommended_action == actual_action,
        }
    ]
)
evaluation_case.to_csv("models/monaco_ver_strategy_evaluation.csv", index=False)

print(result)
print("Reliable offsets:", reliable_offsets)
print("Simulator recommendation:", recommended_action)
print("Actual driver action:", actual_action)
print("Simulator agrees with actual action:", recommended_action == actual_action)
