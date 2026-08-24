"""
app.py
------

Flask web interface for the Smart Queue Waiting Predictor.

Run locally:
    PORT=5001 python src/app.py

Then open:
    http://127.0.0.1:5001

For Render:
    gunicorn src.app:app
"""

import os
import sys
from pathlib import Path

from flask import Flask, render_template, request

# Allow importing predictor.py from the src directory
sys.path.insert(0, str(Path(__file__).parent))

from predictor import load_model_and_metadata, predict_one


# Project root
PROJECT_ROOT = Path(__file__).parent.parent


# Flask application
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates")
)


# Load trained model and metadata
MODEL, META = load_model_and_metadata()


# Preset examples
PRESETS = {
    "worked_example": {
        "num_people_waiting": 12,
        "active_counters": 3,
        "avg_service_time": 7,
        "priority_customers": 2,
        "day_of_week": "Monday",
        "time_of_day": "11:30",
    },
    "zero_waiting": {
        "num_people_waiting": 0,
        "active_counters": 2,
        "avg_service_time": 5,
        "priority_customers": 0,
        "day_of_week": "Wednesday",
        "time_of_day": "10:00",
    },
    "understaffed_peak": {
        "num_people_waiting": 18,
        "active_counters": 1,
        "avg_service_time": 8,
        "priority_customers": 0,
        "day_of_week": "Friday",
        "time_of_day": "18:30",
    },
    "many_priority": {
        "num_people_waiting": 10,
        "active_counters": 3,
        "avg_service_time": 6,
        "priority_customers": 5,
        "day_of_week": "Tuesday",
        "time_of_day": "13:00",
    },
}


# Form field order
FIELD_ORDER = [
    "num_people_waiting",
    "active_counters",
    "avg_service_time",
    "priority_customers",
    "queue_length",
    "day_of_week",
    "time_of_day",
    "historical_queue_load",
    "previous_waiting_time",
]


@app.route("/", methods=["GET", "POST"])
def index():

    # Default empty form
    form_values = {key: "" for key in FIELD_ORDER}

    result = None

    # Handle preset prediction
    preset = request.args.get("preset")

    if preset and preset in PRESETS:
        preset_data = PRESETS[preset]

        form_values.update(
            {key: str(value) for key, value in preset_data.items()}
        )

        result = predict_one(
            preset_data,
            MODEL,
            META
        )

    # Handle normal form submission
    if request.method == "POST":

        raw = {
            key: request.form.get(key, "")
            for key in FIELD_ORDER
        }

        form_values.update(raw)

        result = predict_one(
            raw,
            MODEL,
            META
        )

    return render_template(
        "index.html",
        fields=form_values,
        result=result,
        metrics=META["test_metrics"],
        best_model=META["best_model"],
        days=META["day_of_week_values"],
    )


# Local development server
# Render/Gunicorn will import `app` directly and won't execute this block.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )