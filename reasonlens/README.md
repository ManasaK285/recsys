# ReasonLens

A research platform studying how the *reason given* for a mobile
location-permission request affects whether users grant it, what
precision level they choose, and how much they trust the app and
Android afterward — plus a secondary ML question: does a text-derived
"reason quality" score predict behavior above and beyond the reason's
category?

Full architecture: **Android app (Kotlin/Compose) → FastAPI backend →
SQLite/PostgreSQL → statistical analysis (logistic regression) + ML
reason-quality model → Streamlit dashboard**, all driven by one
reproducible pipeline command.

## Motivation and related work

This project is directly motivated by, and designed to complement,
recent large-scale empirical work on exactly this question:

> Berke, A., Smith, M., Jain, A., & Christodorescu, M. (2027).
> **When Android Apps Request Location with a Reason: A Developer
> Study and User Behavior Experiment.** *Proceedings on Privacy
> Enhancing Technologies*, 2027.

That study surveyed 323 Android developers — most supported a
reason-disclosure requirement, expecting it to improve user privacy and
trust, and even increase grant rates — and ran a randomized controlled
experiment with 2,579 US Android users testing whether app type, the
presence of a reason, and the reason's content (including monetization
framing) affected whether people granted location access. Their
headline finding is important and somewhat counterintuitive: **reason
content did not measurably change users' grant/deny decisions** — those
were driven instead by app type and demographics — but reasons *did*
shift perception of the *platform* positively, so long as the reason
didn't frame the request around advertising.

ReasonLens is built to let a research team run a similarly designed
study end to end — reproducible randomization across app type × reason
type, real Android runtime permission prompts (not a survey vignette),
logged response times, and a post-decision trust/privacy survey — and
to go one step further by testing whether a *continuous, text-derived*
measure of reason quality (specificity, necessity, transparency,
disclosure of advertising use, privacy-protective language) predicts
behavior or trust better than the reason's category alone. Given Berke
et al.'s null result on reason *content* driving decisions, this
platform is equally well suited to a **replication** (same DVs, a new
sample) as to an **extension** (adding the reason-quality ML layer, or
testing whether the platform-trust and ad-framing findings hold with a
different reason taxonomy or app set).

If you use this platform for a study, please cite the paper above, and
consider citing this repository as the tooling used to run it.

```bibtex
@article{berke2027android,
  title   = {When Android Apps Request Location with a Reason: A Developer Study and User Behavior Experiment},
  author  = {Berke, Alex and Smith, May and Jain, Aman and Christodorescu, Mihai},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year    = {2027}
}
```

See `docs/RESEARCH_PLAN.md` for the research questions and design,
`docs/EXPERIMENT_PROTOCOL.md` for the exact participant flow, and
`docs/RESULTS.md` for a sample (simulated) run.

## Quickstart (analysis pipeline + dashboard — no Android device needed)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Runs the entire pipeline in simulation mode: generates 2,000 synthetic
# participants, validates data, computes descriptive stats, runs
# hypothesis tests + logistic regression, trains the ML reason-quality
# model, generates plots, and writes data/results/results.json.
python -m experiments.run_all

# Explore everything interactively
streamlit run dashboard/app.py
```

> On Windows, if `streamlit`/`uvicorn`/`pytest` aren't recognized as
> commands, run them as modules of your Python interpreter instead,
> e.g. `python -m streamlit run dashboard/app.py` — this avoids PATH
> issues with the `Scripts` folder not being on your `PATH`.

Run the test suite:

```bash
pytest
```

## Running the live backend (needed for the Android app or a real pilot)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This exposes:
- `GET  /experiment/config` — the app/reason type lists
- `POST /experiment/assign` — deterministic condition assignment
- `GET  /experiment/new_participant_id` — generate a fresh anonymous participant ID
- `POST /events/permission` — permission-decision logging
- `GET  /events/permission/{participant_id}` — read back a participant's logged events
- `POST /surveys/submit` — trust survey submission
- `GET  /surveys/{participant_id}` — read back a participant's survey responses
- `GET  /health` — health check

Interactive API docs (Swagger UI) are auto-generated at
`http://localhost:8000/docs` — useful for manually exercising the full
assign → log event → submit survey flow before wiring up the Android
app.

By default it uses a local SQLite file
(`backend/reasonlens.db`); set `REASONLENS_DATABASE_URL` to point at
PostgreSQL for production (e.g.
`postgresql://user:pass@host:5432/reasonlens`).

## Running a real pilot

```bash
python -m experiments.run_all --mode pilot --data-dir /path/to/export
```

`--data-dir` must contain `participants.csv`, `assignments.csv`,
`permission_events.csv`, and `survey_responses.csv` exported from the
production database (matching the schemas in `backend/app/models.py`).
The rest of the pipeline (validation → stats → ML → dashboard) is
identical to simulation mode.

## Android app

`android/` is a standard Gradle project — open **the `android/` folder
itself** (not the repo root) in Android Studio and run it on a device
or emulator. It talks to the backend at `http://10.0.2.2:8000/` by
default (the emulator's alias for your host machine's localhost);
change `RetrofitClient.BASE_URL` to your machine's LAN IP for a
physical device, or to a deployed backend's URL in production. It
requests `ACCESS_FINE_LOCATION` / `ACCESS_COARSE_LOCATION` at runtime
and **never reads an actual coordinate** — see `docs/PRIVACY.md`.

> This repository does not include a compiled `.apk`: building one
> requires the Android SDK/Gradle toolchain (Android Studio). Every
> Kotlin source file, the Gradle build scripts, and the manifest are
> included and ready to open.

## Repository layout
android/ Kotlin/Compose research app (source; build with Android Studio)
backend/ FastAPI + SQLAlchemy REST API
experiments/ Config, reproducible randomization, synthetic-data simulator, run_all.py
analysis/ Descriptive stats, hypothesis tests, logistic regression, plots
ml/ Reason-quality dataset, TF-IDF models, evaluation, explainability
dashboard/ Streamlit research dashboard (9 tabs)
data/ synthetic/ processed/ results/ (generated by run_all.py)
docs/ Research plan, protocol, privacy notes, stats methods, results
tests/ pytest suite (18 tests, covering randomization/API/stats/ML)


## Design notes

- **Reproducible randomization**: condition assignment is a pure
  function of `(participant_id, seed)` via SHA-256 hashing, implemented
  identically in the backend (`app/routes/experiment.py`) and the
  simulator (`experiments/randomization.py`), so simulated and pilot
  data are always assignable through the same logic.
- **Simulation vs. pilot are never conflated**: every participant row
  is tagged with `mode`, and the two data sources are kept in separate
  directories (`data/synthetic/` vs. wherever a real export lives).
- **The regression is the primary analysis; ML is a secondary,
  behavioral-analysis component** — not used to inflate scope. See
  `docs/STATISTICAL_METHODS.md`.

## References

Berke, A., Smith, M., Jain, A., & Christodorescu, M. (2027). When
Android Apps Request Location with a Reason: A Developer Study and User
Behavior Experiment. *Proceedings on Privacy Enhancing Technologies*,
2027.