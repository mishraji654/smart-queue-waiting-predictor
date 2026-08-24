"""
predictor.py
------------
Shared prediction core (Assignment Section 10). Both `predict_cli.py` and
the optional `app.py` web demo import this module, so input validation and
defaulting logic live in exactly one place instead of being duplicated.

Handles Section 13's edge cases explicitly:
  - zero people waiting                         -> valid, predicts a small wait
  - one active counter during a busy period      -> valid, no special-casing needed
  - multiple priority customers                  -> valid
  - people waiting far above anything in training -> WARNING, prediction still returned
  - missing / blank optional inputs               -> filled with sensible defaults
  - out-of-range / impossible inputs (0 counters, -ve counts) -> ERROR, no prediction
  - a rare hour/day combination                   -> valid, just note low support in EDA
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import pandas as pd

MODEL_PATH = "models/queue_wait_time_model.pkl"
METADATA_PATH = "models/model_metadata.json"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class PredictionResult:
    ok: bool
    prediction: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    alert: str | None = None
    inputs_used: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


LONG_WAIT_THRESHOLD_MIN = 30  # Optional Extension (Section 16): alert when predicted wait is long


def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return model, meta


def time_period_from_hour(hour: int) -> str:
    if hour <= 5 or hour >= 21:
        return "Night"
    if hour <= 11:
        return "Morning"
    if hour <= 16:
        return "Afternoon"
    return "Evening"


def _to_number(value, field_name, errors, allow_negative=False):
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        errors.append(f"'{field_name}' must be a number (got {value!r}).")
        return None
    if not allow_negative and num < 0:
        errors.append(f"'{field_name}' cannot be negative (got {num}).")
        return None
    return num


def validate_and_fill(raw: dict, meta: dict) -> PredictionResult:
    """raw may contain any subset of the recognised keys, as strings or
    numbers, with missing/optional ones left out or set to None/''."""
    errors: list[str] = []
    warnings: list[str] = []
    ranges = meta["feature_ranges_train"]

    # ---- required fields ----
    num_people_waiting = _to_number(raw.get("num_people_waiting"), "num_people_waiting", errors)
    active_counters = _to_number(raw.get("active_counters"), "active_counters", errors)
    avg_service_time = _to_number(raw.get("avg_service_time"), "avg_service_time", errors)

    if num_people_waiting is None and not any("num_people_waiting" in e for e in errors):
        errors.append("'num_people_waiting' is required.")
    if active_counters is None and not any("active_counters" in e for e in errors):
        errors.append("'active_counters' is required.")
    if avg_service_time is None and not any("avg_service_time" in e for e in errors):
        errors.append("'avg_service_time' is required.")

    if active_counters is not None and active_counters < 1:
        errors.append(f"'active_counters' must be at least 1 (0 counters means nobody can ever be served) - got {active_counters}.")
    if avg_service_time is not None and avg_service_time <= 0:
        errors.append(f"'avg_service_time' must be greater than 0 minutes - got {avg_service_time}.")

    if errors:
        return PredictionResult(ok=False, errors=errors)

    # ---- extreme-but-not-impossible values: warn, don't block ----
    max_people_seen = ranges["num_people_waiting"]["max"]
    if num_people_waiting > max_people_seen:
        warnings.append(
            f"{int(num_people_waiting)} people waiting is above the highest value seen in training "
            f"({int(max_people_seen)}). The prediction is an extrapolation and may be less reliable."
        )
    if active_counters > 6:
        warnings.append(f"{int(active_counters)} active counters is unusually high for this model's training data (max 6 seen); treating it as an extrapolation.")

    # ---- optional fields with defaults ----
    # NOTE: for each optional field we must tell "left blank" (-> use a
    # default) apart from "given but invalid" (-> real error, do not
    # silently paper over it with a default).
    def optional_number(key, default, default_note):
        raw_val = raw.get(key)
        if raw_val in (None, ""):
            warnings.append(default_note.format(default=default))
            return default
        val = _to_number(raw_val, key, errors)
        return val  # may be None if invalid; error already recorded

    priority_customers = optional_number("priority_customers", 0, "'priority_customers' not given - assumed {default} (no priority cases).")

    queue_length = optional_number(
        "queue_length", num_people_waiting + active_counters,
        "'queue_length' not given - estimated as {default:.0f} (people waiting + active counters).",
    )
    if queue_length is not None and queue_length < num_people_waiting:
        warnings.append(f"'queue_length' ({queue_length:.0f}) was less than 'num_people_waiting' ({num_people_waiting:.0f}), which isn't logically consistent - using {num_people_waiting + active_counters:.0f} instead.")
        queue_length = num_people_waiting + active_counters

    historical_queue_load = optional_number(
        "historical_queue_load", ranges["historical_queue_load"]["median"],
        "'historical_queue_load' not given - using the typical training value ({default:.1f}).",
    )
    previous_waiting_time = optional_number(
        "previous_waiting_time", ranges["previous_waiting_time"]["median"],
        "'previous_waiting_time' not given - using the typical training value ({default:.1f}).",
    )

    if errors:
        return PredictionResult(ok=False, errors=errors)

    # ---- day of week ----
    day_raw = (raw.get("day_of_week") or "").strip()
    if not day_raw:
        day_of_week = datetime.now().strftime("%A")
        warnings.append(f"'day_of_week' not given - defaulted to today ({day_of_week}).")
    else:
        match = next((d for d in DAYS if d.lower() == day_raw.lower()), None)
        if match is None:
            errors.append(f"'day_of_week' must be one of {DAYS} - got {day_raw!r}.")
            return PredictionResult(ok=False, errors=errors)
        day_of_week = match

    # ---- time of day ----
    time_raw = (raw.get("time_of_day") or "").strip()
    if not time_raw:
        now = datetime.now()
        hour, minute = now.hour, now.minute
        warnings.append(f"'time_of_day' not given - defaulted to now ({hour:02d}:{minute:02d}).")
    else:
        try:
            hour, minute = map(int, time_raw.split(":"))
            assert 0 <= hour <= 23 and 0 <= minute <= 59
        except (ValueError, AssertionError):
            errors.append(f"'time_of_day' must be in 24-hour HH:MM format (e.g. 14:30) - got {time_raw!r}.")
            return PredictionResult(ok=False, errors=errors)

    if errors:
        return PredictionResult(ok=False, errors=errors)

    time_period = time_period_from_hour(hour)

    clean = {
        "num_people_waiting": num_people_waiting,
        "queue_length": queue_length,
        "active_counters": active_counters,
        "avg_service_time": avg_service_time,
        "priority_customers": priority_customers,
        "day_of_week": day_of_week,
        "time_of_day": f"{hour:02d}:{minute:02d}",
        "historical_queue_load": historical_queue_load,
        "previous_waiting_time": previous_waiting_time,
        "hour": hour,
        "time_period": time_period,
    }
    return PredictionResult(ok=True, inputs_used=clean, warnings=warnings, errors=[])


def predict_one(raw: dict, model=None, meta=None, alert_threshold: float = LONG_WAIT_THRESHOLD_MIN) -> PredictionResult:
    if model is None or meta is None:
        model, meta = load_model_and_metadata()

    result = validate_and_fill(raw, meta)
    if not result.ok:
        return result

    row = pd.DataFrame([{col: result.inputs_used[col] for col in meta["all_features"]}])
    pred = float(model.predict(row)[0])
    pred = max(0.0, pred)  # a queue can't have negative waiting time
    result.prediction = round(pred, 1)

    # Optional Extension (Section 16): prediction range from held-out residual quantiles
    p10, p90 = meta.get("residual_p10"), meta.get("residual_p90")
    if p10 is not None and p90 is not None:
        result.range_low = round(max(0.0, pred + p10), 1)
        result.range_high = round(max(0.0, pred + p90), 1)

    # Optional Extension (Section 16): alert when predicted wait crosses a threshold
    if alert_threshold is not None and pred >= alert_threshold:
        result.alert = f"Long wait predicted ({pred:.0f} min >= {alert_threshold:.0f} min threshold) - consider opening another counter if possible."

    return result
