# ReasonLens — Experiment Protocol

## Participant flow (Android app)

1. **Consent** — participant reads a short study description and taps
   "I agree to participate" or "Decline". Declining ends the session;
   nothing is sent to the backend beyond the fact that consent was not
   given (no participant_id is created).
2. **Assignment** — on consent, the app generates a local anonymous
   `participant_id` (UUID, stored only in local SharedPreferences) and
   calls `POST /experiment/assign`. The backend deterministically maps
   `(participant_id, seed)` to one of 4 app types × 5 reason types (20
   conditions) via SHA-256 hashing, so the same participant always
   reproduces the same assignment for a given experiment seed.
3. **Scenario** — the app displays a short mock screen for the assigned
   app type (e.g. a rideshare "Request ride" screen) to give the
   permission request realistic context.
4. **Reason + permission prompt** — the app shows the reason text for
   the assigned `reason_type`, then triggers the real Android runtime
   permission dialog for `ACCESS_FINE_LOCATION` / `ACCESS_COARSE_LOCATION`.
   The time between the reason text being shown and the OS dialog
   result is recorded as `response_time_ms`.
5. **Permission result logged** — `POST /events/permission` records
   `decision` (granted/denied) and, if granted, `precision`
   (approximate/precise), inferred from which permission Android
   actually granted. **The device's real coordinates are never read,
   requested, or transmitted** — only the binary decision and precision
   level.
6. **Trust survey** — four 5-point Likert items (app trust, Android
   trust, perceived necessity, privacy concern), submitted via
   `POST /surveys/submit`.
7. **Completion** — thank-you screen; the participant may close the app.

## Randomization

Reproducible hash-based assignment (`experiments/randomization.py`):

```
digest = sha256(f"{participant_id}:{seed}")
app_type   = APP_TYPES[int(digest[:8], 16) % len(APP_TYPES)]
reason_type = REASON_TYPES[int(digest[8:16], 16) % len(REASON_TYPES)]
```

This is used identically by the backend's live `/experiment/assign`
endpoint and by `experiments/simulation.py`, so simulated and pilot data
share the exact same randomization logic and can be validated against
each other.

## Attention / data quality checks

The pipeline's `analysis/preprocessing.py::validate()` flags: missing
required fields, invalid decision values, and duplicate participant
rows from an unexpected join. Extend this function with additional
checks (e.g. implausibly fast response times) before running a real
pilot's data through the analysis layer.

## Withdrawal

A participant may withdraw at any point before submitting the survey by
force-closing the app; no partial record beyond already-logged events
is created. Because `participant_id` is anonymous and stored only on
device, there is no mechanism to retroactively delete a specific
participant's server-side rows without the participant sharing their
local ID — this limitation should be disclosed in the consent text
(see `docs/PRIVACY.md`).
