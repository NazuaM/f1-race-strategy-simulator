# F1 Strategy Intelligence

An interactive Formula 1 strategy simulator that compares staying out with
pit-stop options using historical tyre pace, degradation, tyre-life limits, and
Monte Carlo uncertainty.

> **Status:** MVP

## Screenshots

Add screenshots to `docs/screenshots/` using these filenames:

```text
docs/screenshots/strategy-sandbox.png
docs/screenshots/degradation.png
docs/screenshots/ask-the-strategist.png
```

![Strategy sandbox](docs/screenshots/strategy-sandbox.png)

![Degradation by compound](docs/screenshots/degradation.png)

![Ask the strategist](docs/screenshots/ask-the-strategist.png)

## How it works

```text
Hugging Face lap data
          |
          v
src/data.py
  Load and prepare features
          |
          v
src/strategy.py
  Pace offsets, degradation, tyre-life feasibility,
  strategy projections, and Monte Carlo simulation
          |
          +--> app/run_strategy.py
          |
          +--> app/dashboard.py
                    |
                    v
              Optional Groq explanation
```

The simulator calculates the numerical recommendation first. The LLM only
explains the supplied facts; it is not used to calculate strategy totals.

## Run locally

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

The lap data is downloaded at runtime from the Hugging Face dataset configured in
[src/data.py](src/data.py); no dataset files need to be committed to the repository.

## Deploy on Streamlit Community Cloud

Use `app/dashboard.py` as the app file and `requirements.txt` as the dependency
file. Add `GROQ_API_KEY` as a secret to enable the optional strategist panel.
Without that secret, the dashboard still provides the numerical strategy
analysis.

## Groq API key

Never commit an API key to this repository.

For a temporary local PowerShell session:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
python -m streamlit run app\dashboard.py
```

For Streamlit Community Cloud, add `GROQ_API_KEY` in the app's Secrets settings.

The included `.env.example` is a reference template. The current app reads
environment variables directly and does not automatically load `.env` files, so
use PowerShell locally or Streamlit Secrets in deployment.

## Command-line smoke test

```powershell
python app/run_strategy.py
```

The strategy engine:

- anchors projections to the driver's recent current-stint pace;
- removes already accumulated degradation before comparing compounds;
- uses exact-race, historical-circuit, then global degradation rates;
- rejects missing reliable compound offsets instead of silently treating them as zero;
- randomizes compound-specific offsets and degradation rates in Monte Carlo runs.
- marks strategies that exceed circuit-specific historical tyre-life limits as
  infeasible instead of selecting them as winners.

The Monte Carlo output is uncertainty around the model inputs, not a guarantee that
the real driver will choose the simulated winner.

## Current limitations

- The strategy engine currently evaluates one-stint options; it does not yet
  optimize complete multi-stop race plans.
- Historical compound offsets combine available training observations across
  drivers and teams, so they are estimates rather than controlled tyre tests.
- Runtime data is downloaded from the Hugging Face dataset and requires network
  access on first launch.
- The dashboard is a decision-support prototype, not a live race prediction
  system.

## Dataset and credits

This project uses the F1 strategy dataset by Víctor Vega:

- https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset

Please retain the dataset's original license and attribution requirements when
sharing or extending this project.
