# HydroTwin AI — Digital Twin for Water Distribution Networks with GenAI Decision Support

An AI-driven digital twin of a municipal water distribution network (WDN). It couples a physics-based hydraulic simulator (EPANET / WNTR) with two machine-learning models — **leak detection & localization** and **24-hour water demand forecasting** — and a **GenAI operator assistant** that turns live telemetry into incident triage and containment steps.

> Academic project — School of Computer Science & Engineering (SCOPE), VIT.
> Team: Azim Abdulla, Adithya Shaji, Noel John · Supervisor: Dr. Archana T

---

## What it does

| Capability | How it works | Output |
|---|---|---|
| **Hydraulic twin** | WNTR pressure-dependent-demand (PDD) simulation of EPANET **Net1** over 24 hourly steps | Node pressures, pipe flows, tank levels |
| **Demand forecasting** | `HistGradientBoostingRegressor` trained on the **Battle of Water Demand Forecasting (BWDF)** dataset; forecast drives the twin's demand pattern | 24 h demand (L/s) with 95% band, peak / minimum-night-flow hours |
| **Leak injection** | Emitter leak added at any junction from hour 4, with configurable orifice area | Perturbed pressures and leak outflow |
| **Leak detection & localization** | 100-tree `RandomForestClassifier` on 18 sensor features (9 pressures + 9 deltas vs baseline) | Leak probability, localized junction, severity, top affected sensors |
| **Leak vs. demand-surge disambiguation** | Separates structural breaches from weather-driven consumption peaks | Status, category, recommended action |
| **24 h risk lookahead** | Scans the simulated horizon for low-pressure events and Tank 2 depletion | Min pressure node/hour, tank reserve %, event count |
| **GenAI operator assistant** | Telemetry-grounded prompt to Google Gemini, with a deterministic offline expert fallback; every exchange logged to SQLite | Markdown incident report + structured advisory |

### Operator UI (React)

Three views share one scenario state (temperature, weekday/weekend, simulation hour):

1. **Hydraulic Twin** — interactive network map, pressure time-series per node, leak control panel, live ML alert card.
2. **Demand Forecasting** — 24 h forecast with confidence band and risk assessment.
3. **GenAI Assistant** — chat interface that sends the current twin state with each question.

Scenario presets: *Nominal Baseline (22 °C)*, *Summer Heatwave (36 °C)*, *Cool Rainy Day (14 °C)*, *Weekend Peak (26 °C)*, plus a custom temperature slider.

---

## Architecture

```
┌──────────────────────── Frontend (React 19 + Vite + Tailwind + Recharts) ───────────────────────┐
│  Hydraulic Twin view        Demand Forecasting view        GenAI Assistant view                 │
└───────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                            │ REST (http://localhost:8000)
┌───────────────────────────────────────────▼──────────────────────────────────────────────────────┐
│                                  FastAPI backend (app.py)                                        │
│                                                                                                  │
│  demand_forecast/ ──► 24h demand ──► simulation.py (WNTR PDD on Net1.inp) ──► pressures/flows    │
│        (BWDF HGB)       multipliers          │  + optional leak injection                        │
│                                              ├──► risk assessment + disambiguation               │
│                                              ▼                                                   │
│                              leak_detection/ (Random Forest) ──► leak alert                      │
│                                              │                                                   │
│                                              ▼                                                   │
│                          genai/advisor.py (Gemini ↔ offline expert) ──► SQLite incident log      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow for a leak scenario:** forecast demand for the chosen weather → rescale Net1's demand pattern → simulate with and without the leak → feed hour-12 pressures and deltas to the Random Forest → attach risk and disambiguation → the frontend re-runs detection as the operator scrubs through hours → the operator asks the assistant, which receives the full state.

---

## Repository structure

```
digital-twin-wdn-genai/
├── backend/
│   ├── app.py                     # FastAPI app and endpoints
│   ├── simulation.py              # WNTR simulation, leak injection, risk, disambiguation
│   ├── ai.py                      # Backward-compatible wrapper over leak_detection
│   ├── leak_detection/
│   │   ├── dataset_generator.py   # Synthetic 5k-sample dataset from WNTR runs
│   │   ├── train_model.py         # Random Forest training (80/20 stratified)
│   │   ├── evaluate_model.py      # Classification report, confusion matrix, importances
│   │   └── inference.py           # predict_leak(): probability, node, severity
│   ├── demand_forecast/
│   │   ├── data_loader.py         # BWDF Excel → cleaned hourly CSV
│   │   ├── feature_engineering.py # Lags (1/2/24/48/168h), rolling stats, cyclical time, weather
│   │   ├── train_forecast.py      # HGB regressor, chronological 80/20 split
│   │   ├── evaluate_forecast.py
│   │   └── inference.py           # 24h forecast scaled and distributed to Net1 junctions
│   ├── genai/
│   │   ├── advisor.py             # Prompt construction, Gemini calls, offline fallback
│   │   └── incident_logger.py     # SQLite incident_logs table
│   ├── notebooks/
│   │   ├── 01_leak_detection_and_localization.ipynb
│   │   └── 02_water_demand_forecasting.ipynb
│   ├── data/
│   │   ├── Net1.inp                       # EPANET Net1 network
│   │   ├── leak_dataset.csv               # Generated training set (~5k rows)
│   │   ├── bwdf_demand_cleaned.csv        # 19,683 hourly records, Jan 2021 – Mar 2023
│   │   ├── leak_model.joblib              # Trained Random Forest bundle
│   │   └── demand_forecast_model.joblib   # Trained HGB bundle
│   ├── requirements.txt
│   └── .env.example
├── bwdf/                          # Raw BWDF benchmark files (inflow, weather, reference solutions, paper)
└── frontend/
    └── src/
        ├── App.tsx                # Shared scenario state, view routing
        ├── api.ts                 # Backend client
        ├── types.ts
        ├── views/                 # HydraulicTwin, DemandForecasting, GenAIAssistant
        └── components/            # NetworkMap, PressureChart, LeakControlPanel, AIAlertCard, …
```

---

## Models

### Leak detection & localization

| Item | Detail |
|---|---|
| Network | EPANET Net1 — 9 junctions (`10, 11, 12, 13, 21, 22, 23, 31, 32`), 1 tank, 1 reservoir, 1 pump |
| Training data | 5,000 WNTR-generated samples: 30% normal (demand factor 0.7–1.3×), 70% leaks across all 9 junctions at 5 orifice sizes (0.001–0.012 m², ±20% jitter), Gaussian sensor noise σ = 0.4 m |
| Features (18) | `P_j` (current pressure) and `dP_j` (baseline − current) for each junction |
| Classes (10) | One per junction + `Normal` |
| Model | `RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=4)` |
| Test accuracy | **93.1%** on a 20% stratified hold-out (999 samples), per the committed `leak_model.joblib` |
| Decision rule | Leak if top class ≠ `Normal` and P(leak) > 0.40. Severity escalates to HIGH / CRITICAL with probability (0.65 / 0.85) or max pressure drop (12 m / 25 m) |

### Demand forecasting

| Item | Detail |
|---|---|
| Data | BWDF — inflows from 10 Italian DMAs + weather (rain, temperature, humidity, wind), 19,683 hours |
| Target | Total network demand (sum of DMAs, L/s) |
| Features (18) | Cyclical hour / day-of-week, weekend flag, month, weather, lags (1, 2, 24, 48, 168 h), rolling mean (6 h, 24 h), rolling std (24 h) |
| Model | `HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8)` |
| Validation | Chronological 80/20 split — train to Oct 2022, test Oct 2022 – Mar 2023 |
| Test metrics | **R² 0.981 · MAE 4.14 L/s · RMSE 5.49 L/s · MAPE 1.72%** |
| Twin coupling | Forecast scaled (×0.22) to Net1's ~69 L/s nominal demand and written into EPANET pattern `1` as hourly multipliers |

### GenAI assistant

- Builds a telemetry prompt with simulation hour, temperature, day type, the Random Forest verdict, Tank 2 reserve, minimum network pressure and all junction pressures, followed by the operator's question.
- Uses the `google-genai` SDK with a system instruction that frames the model as a hydraulic operations and incident-response assistant (temperature 0.3). Tries several Gemini Flash model IDs in order.
- With no API key, or if every call fails, returns a deterministic **offline expert** report for one of three cases: confirmed breach, heatwave demand surge, or nominal.
- Every response is written to `backend/data/digital_twin_incidents.db` and can be read back from `GET /ai-incidents`.

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 20+ (Vite 8 / React 19)
- Optional: a Google Gemini API key

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
# Not yet listed in requirements.txt but required:
pip install scikit-learn joblib google-genai openpyxl

cp .env.example .env        # optionally set GEMINI_API_KEY=...
python app.py               # → http://localhost:8000  (Swagger UI at /docs)
```

### Frontend

```bash
cd frontend
npm install
npm run dev                 # → http://localhost:5173
```

To point the UI at a different backend, set `VITE_API_URL` (default `http://localhost:8000`).

### Retraining the models

The committed `.joblib` files are pickled with a specific scikit-learn version and may not load under a different one. If loading fails, retrain from `backend/`:

```bash
python -m demand_forecast.train_forecast     # uses data/bwdf_demand_cleaned.csv (rebuilds it from ../bwdf/*.xlsx if missing)
python -m leak_detection.train_model         # uses data/leak_dataset.csv (regenerates via WNTR if missing)
python -m leak_detection.evaluate_model      # full classification report + confusion matrix
```

If a model file is missing, both inference modules train it automatically on first request.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service status and endpoint index |
| GET | `/network` | Nodes (type, elevation, base demand, x/y) and links (length, diameter in mm) |
| GET | `/simulate?temp=24&is_weekend=0` | Baseline 24 h simulation with forecast, risk assessment, disambiguation and ML check |
| POST | `/inject-leak` | `{node_id, leak_area, temperature, is_weekend}` → leak simulation and ML detection versus same-weather baseline |
| POST | `/detect-leak` | `{current_pressures, baseline_pressures?}` → Random Forest inference on any pressure vector |
| GET | `/forecast-demand?temp=24&is_weekend=0` | 24 h demand forecast with 95% band and junction-level split |
| POST | `/ask-ai` | `{prompt, current_timestep, temperature, is_weekend, leak_node_id, current_pressures, ai_alert, risk_assessment, disambiguation}` → assistant response |
| GET | `/ai-incidents?limit=20` | Recent advisory history from SQLite |

Example:

```bash
curl -X POST localhost:8000/inject-leak \
  -H "Content-Type: application/json" \
  -d '{"node_id":"22","leak_area":0.008,"temperature":24,"is_weekend":0}'
```

---

## Known limitations

These are worth knowing before you extend the prototype or rely on its outputs.

- **Disambiguation is rule-based, not inferred.** `simulation.py` labels a run as a breach when a leak was injected, and as a demand surge when temperature ≥ 30 °C. Its confidence values (0.96 / 0.94 / 0.98) are fixed. The Random Forest is the only learned detector.
- **Forecast inference uses synthetic history.** At runtime the lag and rolling features are built from a diurnal template and the input temperature, not from observed recent demand. The output is a weather-conditioned profile rather than a true one-step-ahead forecast.
- **Train/serve mismatch in leak detection.** Training data uses Net1's default demand pattern, but the live twin overwrites that pattern with forecast-driven multipliers. Accuracy under unusual scenarios such as heatwaves is likely lower than the hold-out figure.
- **Single small network.** Everything is calibrated to Net1 (9 junctions, one sensor per junction). Scaling to a real utility network means sparse sensors, many more candidate leak locations and calibrated demand allocation.
- **GenAI structured output is templated.** In the Gemini path, `structured_advisory.steps` are fixed strings; only `response_text` comes from the model. The offline report hard-codes asset names (Pump 9, Tank 2, J31/J32) and some figures.
- **Reported vs. measured accuracy.** The API root advertises 97.4% leak-detection accuracy; the committed model bundle records 93.1% test accuracy.
- **Dependencies.** `requirements.txt` omits `scikit-learn`, `joblib`, `google-genai` and `openpyxl`, and no versions are pinned.
- **Prototype security posture.** CORS allows all origins, there is no authentication, and SQLite is the only persistence layer.

## Suggested next steps

1. Pin dependencies, including the scikit-learn version the models were serialized with, and add the missing packages.
2. Replace the rule-based disambiguation with a residual test: observed inflow versus the forecast band, combined with how localized the pressure drop is.
3. Generate leak-training data across forecast-driven demand scenarios to remove the train/serve mismatch.
4. Ask Gemini for JSON output (severity, root cause, steps) and validate it, instead of templating `structured_advisory`.
5. Evaluate on a larger benchmark network (e.g. L-Town / BattLeDIM) with a limited sensor budget.

---

## Data sources & acknowledgements

- **EPANET Net1** — standard example network distributed with EPANET (US EPA).
- **WNTR** — Water Network Tool for Resilience, Sandia National Laboratories / US EPA.
- **BWDF** — *Battle of Water Demand Forecasting* (WDSA/CCWI 2024) dataset and supplemental materials (Alvisi et al.), included under `bwdf/`. Check the dataset's original terms before redistributing.
