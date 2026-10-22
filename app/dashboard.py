import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import test_dataset, train_dataset, train_dataset_sorted  # noqa: E402
from src.strategy import (  # noqa: E402
    build_race_summary,
    build_stint_table,
    calculate_historical_pace_offsets,
    calculate_stint_life_limits,
    current_pace_anchor,
    evaluate_all_strategies,
    get_degradation_rate,
    run_monte_carlo_strategy,
)


@st.cache_resource
def load_strategy_context():
    stint_table = build_stint_table(train_dataset_sorted)
    race_summary, overall_compound = build_race_summary(stint_table)
    return race_summary, overall_compound


def facts_prompt(facts, question):
    return (
        "You are an F1 race strategist. Explain only the supplied facts. "
        "Do not invent telemetry or change the recommendation. Mention uncertainty "
        "when probabilities are not decisive.\n\n"
        f"Facts:\n{json.dumps(facts, indent=2)}\n\n"
        f"Question: {question}"
    )


def explain_with_llm(facts, question):
    """Explain the given facts using the LLM."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "LLM explanation is disabled. Set GROQ_API_KEY in the "
            "deployment environment to enable the strategist."
        )

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b",
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an F1 race strategist. Explain only the "
                        "supplied simulator facts. Do not invent telemetry, "
                        "change numerical values, or create strategies that "
                        "are marked infeasible. If no feasible one-stint "
                        "strategy exists, say so clearly. Keep the answer "
                        "under 250 words and use short paragraphs."
                    ),
                },
                {
                    "role": "user",
                    "content": facts_prompt(facts, question),
                },
            ],
            temperature=0.2,
            max_tokens=800,
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"LLM request failed: {error}"


st.set_page_config(page_title="F1 Strategy Intelligence", page_icon="🏎️", layout="wide")
st.title("F1 Strategy Intelligence")
st.caption("Data-driven pace, degradation, and pit-window analysis")

race_summary, overall_compound = load_strategy_context()

with st.sidebar:
    st.header("Race setup")
    races = sorted(test_dataset["GP_Name"].dropna().unique())
    gp_name = st.selectbox("Race", races)
    race = test_dataset[test_dataset["GP_Name"] == gp_name]
    drivers = sorted(race["Driver"].dropna().unique())
    driver = st.selectbox("Driver", drivers)
    driver_laps = race[race["Driver"] == driver].sort_values("LapNumber")
    max_lap = int(driver_laps["LapNumber"].max())
    decision_lap = st.slider("Decision lap", 1, max_lap, min(max_lap // 2, max_lap))
    current_row = driver_laps[driver_laps["LapNumber"] <= decision_lap].tail(1)
    if current_row.empty:
        st.warning("No driver data is available at this decision lap.")
        st.stop()
    current_compound = st.selectbox(
        "Current compound",
        [str(current_row["Compound"].iloc[0])],
    )
    current_tyre_life = current_row["TyreLife"].iloc[0]
    if pd.isna(current_tyre_life):
        st.error("Tyre age is unavailable for this decision lap.")
        st.stop()
    current_age = st.number_input(
        "Current tyre age",
        min_value=1,
        max_value=100,
        value=int(current_tyre_life),
    )
    default_laps_remaining = current_row["laps_remaining"].iloc[0]
    if pd.isna(default_laps_remaining):
        default_laps_remaining = 1
    laps_remaining = st.number_input(
        "Laps remaining",
        min_value=1,
        max_value=100,
        value=max(1, int(default_laps_remaining)),
    )
    pit_loss = st.number_input("Pit loss (seconds)", min_value=10.0, max_value=40.0, value=22.0)
    simulations = st.number_input("Monte Carlo simulations", 100, 10000, 1000, 100)

try:
    pace_now, recent_age = current_pace_anchor(
        test_dataset, 2025, gp_name, driver, decision_lap
    )
    offsets_table = calculate_historical_pace_offsets(train_dataset, gp_name)
    reliable_offsets = offsets_table[
        offsets_table["Reliable"]
    ]["OffsetFromMedium"].to_dict()
    available_compounds = list(reliable_offsets)
    stint_life_limits = calculate_stint_life_limits(train_dataset, gp_name)
    if current_compound not in reliable_offsets:
        st.error(f"No reliable historical offset exists for {current_compound}.")
        st.stop()

    result = evaluate_all_strategies(
        year=2025,
        gp_name=gp_name,
        current_compound=current_compound,
        current_age=current_age,
        laps_remaining=laps_remaining,
        baseline_pace=pace_now,
        pit_loss=pit_loss,
        dataset=test_dataset,
        race_summary=race_summary,
        overall_compound=overall_compound,
        pace_offsets=reliable_offsets,
        available_compounds=available_compounds,
        recent_age=recent_age,
        stint_life_limits=stint_life_limits,
    )
    probabilities = run_monte_carlo_strategy(
        year=2025,
        gp_name=gp_name,
        current_compound=current_compound,
        current_age=current_age,
        laps_remaining=laps_remaining,
        baseline_pace=pace_now,
        pit_loss=pit_loss,
        simulations=int(simulations),
        dataset=test_dataset,
        race_summary=race_summary,
        overall_compound=overall_compound,
        pace_offsets=reliable_offsets,
        available_compounds=available_compounds,
        recent_age=recent_age,
        stint_life_limits=stint_life_limits,
    )
except (ValueError, KeyError) as error:
    st.error(str(error))
    st.stop()

tab_overview, tab_degradation, tab_strategy, tab_llm = st.tabs(
    ["Race overview", "Degradation", "Strategy sandbox", "Ask the strategist"]
)

with tab_overview:
    st.subheader(f"{driver} at {gp_name}")
    chart = driver_laps[["LapNumber", "FuelAdjustedLapTime", "TyreLife"]].set_index(
        "LapNumber"
    )
    st.line_chart(chart[["FuelAdjustedLapTime"]])
    st.caption("Fuel-adjusted lap time by lap. Tyre age is available in the table below.")
    st.dataframe(driver_laps[["LapNumber", "Compound", "TyreLife", "Position"]].tail(20))

with tab_degradation:
    st.subheader("Historical degradation")
    st.dataframe(
        race_summary[
            (race_summary["GP_Name"] == gp_name)
            & race_summary["Compound"].isin(available_compounds)
        ][["Year", "Compound", "Stints", "StrategyData", "UseRaceData"]]
    )
    st.bar_chart(
        pd.Series(
            {
                compound: get_degradation_rate(
                    2025, gp_name, compound, race_summary, overall_compound
                )
                for compound in available_compounds
            },
            name="seconds per tyre lap",
        )
    )

with tab_strategy:
    st.subheader("Projected strategy")
    if not result["Feasible"].any():
        st.warning(
            "No one-stint strategy fits the estimated tyre-life limits. "
            "A multi-stint strategy model is needed for this scenario."
        )
    st.dataframe(result, use_container_width=True)
    st.caption(f"Estimated tyre-life limits: {stint_life_limits}")
    st.bar_chart(pd.Series(probabilities, name="win probability (%)"))
    st.json(probabilities)

with tab_llm:
    st.subheader("Ask the strategist")
    question = st.text_input("Question", "Should I pit now?")
    if st.button("Explain strategy"):
        facts = {
            "race": {"year": 2025, "circuit": gp_name, "driver": driver},
            "decision": {
                "lap": decision_lap,
                "current_compound": current_compound,
                "current_tyre_age": current_age,
                "laps_remaining": laps_remaining,
                "pace_now": pace_now,
                "recent_age": recent_age,
                "pit_loss": pit_loss,
                "stint_life_limits": stint_life_limits,
            },
            "tyre_rates": {
                compound: get_degradation_rate(
                    2025, gp_name, compound, race_summary, overall_compound
                )
                for compound in available_compounds
            },
            "strategy_table": result.astype(object)
            .where(result.notna(), None)
            .to_dict(orient="records"),
            "monte_carlo_probabilities": probabilities,
        }
        st.write(explain_with_llm(facts, question))
