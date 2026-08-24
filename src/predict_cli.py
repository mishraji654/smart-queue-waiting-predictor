"""
predict_cli.py
---------------
Command-line Prediction Interface (Assignment Section 10).

Three ways to use it:

  1. Interactive (just run it, no arguments) - prompts for each queue
     condition, validates as you go, re-asks on bad input, lets you press
     Enter to accept a sensible default for the optional fields.

        python src/predict_cli.py

  2. One-shot via flags - for scripting / quick checks:

        python src/predict_cli.py --people 12 --counters 3 --service-time 7 --priority 2 --day Monday --time 11:30

  3. Demo mode - runs the model against a handful of built-in scenarios,
     including every edge case from Assignment Section 13:

        python src/predict_cli.py --demo
"""

import argparse
import sys

from predictor import load_model_and_metadata, predict_one


def ask(prompt_text, required=True, cast=str):
    while True:
        raw = input(prompt_text).strip()
        if raw == "" and not required:
            return None
        if raw == "" and required:
            print("  This field is required - please enter a value.")
            continue
        return raw


def run_interactive(model, meta):
    print("=" * 62)
    print("  Smart Queue Waiting-Time Predictor")
    print("=" * 62)
    print("Enter the current queue conditions. Press Enter to skip any")
    print("optional field (marked [optional]) and use a sensible default.\n")

    raw = {}
    while True:
        raw["num_people_waiting"] = ask("People currently waiting ahead: ")
        raw["active_counters"] = ask("Active counters open: ")
        raw["avg_service_time"] = ask("Average service time per person (minutes): ")
        raw["priority_customers"] = ask("Priority customers ahead [optional]: ", required=False)
        raw["queue_length"] = ask("Total queue length [optional]: ", required=False)
        raw["day_of_week"] = ask("Day of week (e.g. Monday) [optional, defaults to today]: ", required=False)
        raw["time_of_day"] = ask("Time now, 24h HH:MM (e.g. 14:30) [optional, defaults to now]: ", required=False)
        raw["historical_queue_load"] = ask("Recent historical queue load [optional]: ", required=False)
        raw["previous_waiting_time"] = ask("Previous customer's waiting time (minutes) [optional]: ", required=False)

        result = predict_one(raw, model, meta)

        if not result.ok:
            print("\nCouldn't produce a prediction because of the following:")
            for e in result.errors:
                print(f"  - {e}")
            print("Please re-enter the values below.\n")
            continue

        print_result(result)

        again = ask("\nPredict another? (y/n): ", required=False)
        if not again or again.lower().startswith("n"):
            print("Goodbye!")
            break
        print()


def print_result(result):
    print(f"\n>>> Estimated waiting time: {result.prediction:.0f} minutes  ({result.prediction:.1f} min precise)")
    if result.range_low is not None:
        print(f"    Likely range: {result.range_low:.0f}-{result.range_high:.0f} minutes")
    if result.alert:
        print(f"    ALERT: {result.alert}")
    if result.warnings:
        print("\nNotes:")
        for w in result.warnings:
            print(f"  - {w}")


def run_demo(model, meta):
    scenarios = [
        ("Assignment's worked example (12 waiting, 3 counters, 7 min service, 2 priority)",
         {"num_people_waiting": 12, "active_counters": 3, "avg_service_time": 7, "priority_customers": 2, "day_of_week": "Monday", "time_of_day": "11:30"}),
        ("Edge case: zero people waiting",
         {"num_people_waiting": 0, "active_counters": 2, "avg_service_time": 5, "day_of_week": "Wednesday", "time_of_day": "10:00"}),
        ("Edge case: only one counter open during a busy evening",
         {"num_people_waiting": 18, "active_counters": 1, "avg_service_time": 8, "day_of_week": "Friday", "time_of_day": "18:30"}),
        ("Edge case: multiple priority customers",
         {"num_people_waiting": 10, "active_counters": 3, "avg_service_time": 6, "priority_customers": 5, "day_of_week": "Tuesday", "time_of_day": "13:00"}),
        ("Edge case: unusually high queue vs training data (extrapolation)",
         {"num_people_waiting": 150, "active_counters": 3, "avg_service_time": 6, "day_of_week": "Saturday", "time_of_day": "12:00"}),
        ("Edge case: missing/incomplete input (only the 3 required fields given)",
         {"num_people_waiting": 9, "active_counters": 2, "avg_service_time": 7}),
        ("Edge case: rare time period (3 AM)",
         {"num_people_waiting": 4, "active_counters": 1, "avg_service_time": 6, "day_of_week": "Sunday", "time_of_day": "03:15"}),
        ("Quiet weekday morning, plenty of staff",
         {"num_people_waiting": 3, "active_counters": 5, "avg_service_time": 4, "day_of_week": "Wednesday", "time_of_day": "09:00"}),
    ]

    print("=" * 70)
    print("  DEMO MODE - running the model against several example scenarios")
    print("=" * 70)

    for title, raw in scenarios:
        print(f"\n--- {title} ---")
        print(f"Input: {raw}")
        result = predict_one(raw, model, meta)
        if result.ok:
            print(f"Estimated waiting time: {result.prediction:.0f} minutes")
            for w in result.warnings:
                print(f"  note: {w}")
        else:
            print("Rejected:")
            for e in result.errors:
                print(f"  - {e}")

    print("\n" + "=" * 70)
    print("Note: the assignment's own worked example (18 minutes) is just a ")
    print("format illustration, not a target this model was built to reproduce -")
    print("its own dataset/model naturally gives a different number for those ")
    print("same inputs. See REPORT.md, Section 'Candidate Questions', for why.")

    # Optional Extension (Section 16): compare predictions before/after adding a counter
    print("\n" + "=" * 70)
    print("  BONUS: effect of opening one more counter (same busy scenario)")
    print("=" * 70)
    base = {"num_people_waiting": 18, "active_counters": 1, "avg_service_time": 8, "day_of_week": "Friday", "time_of_day": "18:30"}
    for counters in (1, 2, 3):
        scenario = dict(base, active_counters=counters)
        r = predict_one(scenario, model, meta)
        print(f"  {counters} counter(s) open -> {r.prediction:.0f} min predicted wait")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Smart Queue Waiting-Time Predictor - CLI")
    p.add_argument("--demo", action="store_true", help="run built-in example/edge-case scenarios and exit")
    p.add_argument("--people", type=str, help="number of people waiting")
    p.add_argument("--counters", type=str, help="active counters")
    p.add_argument("--service-time", type=str, help="average service time (minutes)")
    p.add_argument("--priority", type=str, help="priority customers [optional]")
    p.add_argument("--queue-length", type=str, help="total queue length [optional]")
    p.add_argument("--day", type=str, help="day of week [optional]")
    p.add_argument("--time", type=str, help="time of day HH:MM [optional]")
    p.add_argument("--historical-load", type=str, help="historical queue load [optional]")
    p.add_argument("--previous-wait", type=str, help="previous customer's wait [optional]")
    return p


def main():
    args = build_arg_parser().parse_args()
    try:
        model, meta = load_model_and_metadata()
    except FileNotFoundError:
        print("Model not found. Run the pipeline first:")
        print("  python src/generate_data.py && python src/preprocess.py && python src/train.py")
        sys.exit(1)

    if args.demo:
        run_demo(model, meta)
        return

    # one-shot mode if any flag beyond --demo was given
    if args.people or args.counters or args.service_time:
        raw = {
            "num_people_waiting": args.people,
            "active_counters": args.counters,
            "avg_service_time": args.service_time,
            "priority_customers": args.priority,
            "queue_length": args.queue_length,
            "day_of_week": args.day,
            "time_of_day": args.time,
            "historical_queue_load": args.historical_load,
            "previous_waiting_time": args.previous_wait,
        }
        result = predict_one(raw, model, meta)
        if result.ok:
            print_result(result)
        else:
            print("Couldn't produce a prediction:")
            for e in result.errors:
                print(f"  - {e}")
            sys.exit(1)
        return

    run_interactive(model, meta)


if __name__ == "__main__":
    main()
