"""
run_pipeline.py
----------------
Runs the whole project pipeline in order, so you don't have to remember
the individual commands. Equivalent to running, one after another:

    python src/generate_data.py
    python src/preprocess.py
    python src/eda.py
    python src/train.py

Run:
    python run_pipeline.py
"""

import subprocess
import sys

STEPS = [
    ("Generating synthetic dataset", ["src/generate_data.py"]),
    ("Cleaning data & feature engineering", ["src/preprocess.py"]),
    ("Running exploratory data analysis", ["src/eda.py"]),
    ("Training & evaluating models", ["src/train.py"]),
]


def main():
    for title, cmd in STEPS:
        print("\n" + "=" * 70)
        print(f"STEP: {title}")
        print("=" * 70)
        result = subprocess.run([sys.executable] + cmd)
        if result.returncode != 0:
            print(f"\nStopped: '{' '.join(cmd)}' failed (exit code {result.returncode}).")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)
    print("Try a prediction:")
    print("  python src/predict_cli.py --demo")
    print("  python src/predict_cli.py")
    print("  python src/app.py            (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    main()
