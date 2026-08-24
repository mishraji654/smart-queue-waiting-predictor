# Smart Queue Waiting-Time Predictor

A complete, working machine-learning solution for the **Smart Queue Waiting-Time Predictor** assignment: it estimates how long a person will wait in a queue (hospital, restaurant, bank, government office, customer service center) based on the current queue conditions.

This project is a full ML workflow, not just a model file:
**synthetic data generation → data cleaning → EDA → model training/comparison → evaluation → feature interpretation → a runnable prediction interface.**

Everything in this folder actually runs end-to-end and was tested before being handed to you — the `data/`, `models/` and `outputs/` folders already contain the generated dataset, the trained model, and all plots/reports, so you can look at the results immediately **and** regenerate everything yourself from scratch.

---

## 1. What's inside

```
smart-queue-waiting-predictor/
├── README.md                    <- you are here (setup + how to run)
├── REPORT.md                    <- write-up: methodology, results, answers to the assignment's questions
├── requirements.txt              <- Python dependencies
├── run_pipeline.py                <- runs the whole pipeline in one command
│
├── src/
│   ├── generate_data.py          <- Step 1: creates the synthetic dataset
│   ├── preprocess.py             <- Step 2: cleans data, engineers features, splits train/test
│   ├── eda.py                    <- Step 3: exploratory analysis, saves charts
│   ├── train.py                  <- Step 4: trains + compares models, saves the best one
│   ├── predictor.py              <- shared prediction logic (validation, defaults)
│   ├── predict_cli.py            <- prediction interface #1: terminal / command line
│   └── app.py                    <- prediction interface #2 (bonus): local web form
│
├── templates/index.html          <- HTML for the web form (used by app.py)
│
├── data/
│   ├── raw_queue_data.csv        <- generated dataset (with intentional data-quality issues)
│   ├── cleaned_queue_data.csv    <- after cleaning + feature engineering
│   ├── train.csv / test.csv      <- 80/20 split used for modeling
│
├── models/
│   ├── queue_wait_time_model.pkl <- the trained, saved model (ready to use)
│   └── model_metadata.json       <- feature list, metrics, training data ranges
│
└── outputs/
    ├── data_quality_report.md    <- what was wrong with the raw data + how it was fixed
    ├── eda_summary.md            <- written summary of the exploratory analysis
    ├── training_report.md        <- model comparison, chosen model, feature importance, leakage check
    ├── model_comparison.csv      <- all model scores in one table
    └── plots/                    <- all 9 charts (distributions, comparisons, importance, etc.)
```

## 2. What the project actually does (short version)

1. **`generate_data.py`** simulates ~4,000 realistic queue records (a public dataset with this exact feature set doesn't really exist, and the assignment explicitly allows a synthetic one). It deliberately injects missing values, duplicate rows, and impossible values (0 counters, negative wait times) so the cleaning step has real work to do — see `REPORT.md` for why this is a defensible way to build the dataset.
2. **`preprocess.py`** audits and cleans that data, engineers a couple of time-based features, checks for target leakage, and splits it into train/test sets.
3. **`eda.py`** produces 7 charts answering the assignment's exploration questions (distribution, day/time patterns, queue size vs. wait, counters vs. wait, correlations, outliers).
4. **`train.py`** trains a **Linear Regression**, **Random Forest**, and **Gradient Boosting** model, compares all of them (with 5-fold cross-validation) against **two baselines** (a naive median guess and a simple business-logic formula), picks the best one, explains it with permutation importance, and runs an ablation test to double-check the leakage question empirically.
5. **`predict_cli.py`** / **`app.py`** let you actually type in queue conditions and get a prediction, with input validation and edge-case handling.

Full reasoning, results, and answers to the assignment's "Candidate Questions" are in **`REPORT.md`** — that's the file to read/submit alongside the code if this needs to be written up.

---

## 3. Setup (do this once)

### Prerequisites
- **Python 3.10 or newer**. Check your version:
  ```bash
  python3 --version
  ```
  (On Windows this may just be `python --version`.) If you're below 3.10, install a recent Python from [python.org](https://www.python.org/downloads/) first.

### Step-by-step

**1. Unzip the project** and open a terminal inside the `smart-queue-waiting-predictor` folder (the folder containing this README).

**2. (Recommended) Create a virtual environment** — keeps these packages separate from the rest of your system:

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (Command Prompt):
```bat
python -m venv venv
venv\Scripts\activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked because your terminal prompt will show `(venv)` at the start.

**3. Install the dependencies:**
```bash
pip install -r requirements.txt
```

That's the entire setup. No API keys, no external services, no internet access needed once the packages are installed.

---

## 4. How to run it

### Option A — Everything's already generated, just try a prediction

The repo ships with the dataset, trained model, and plots already built, so you can skip straight to:

```bash
python src/predict_cli.py --demo
```

This runs the model against 8 example scenarios (including every edge case from the assignment) and prints predictions for each — a quick way to see it work with no typing required.

For an interactive session where you type in your own numbers:
```bash
python src/predict_cli.py
```

For the (optional, bonus) browser version:
```bash
python src/app.py
```
then open **http://127.0.0.1:5000** in your browser. Press `Ctrl+C` in the terminal to stop it when you're done.

### Option B — Regenerate everything from scratch

If you want to see/verify the full pipeline run yourself (or re-run it with different random data), either run the one convenience command:

```bash
python run_pipeline.py
```

or run each stage individually, in this order:

```bash
python src/generate_data.py     # creates data/raw_queue_data.csv
python src/preprocess.py        # creates data/cleaned_queue_data.csv, train.csv, test.csv
python src/eda.py               # creates the charts in outputs/plots/
python src/train.py             # trains models, saves the best one to models/
```

Each script prints a readable summary as it runs, and also saves that summary to a `.md` file in `outputs/` so you don't have to re-run anything to see the results again later.

After that, use `predict_cli.py` or `app.py` exactly as in Option A — they'll now be using your freshly retrained model.

---

## 5. Using the prediction interface

### Command line — one-shot flags
```bash
python src/predict_cli.py --people 12 --counters 3 --service-time 7 --priority 2 --day Monday --time 11:30
```
Only `--people`, `--counters`, and `--service-time` are required; everything else is optional and will use a sensible default if left out (shown to you as a note in the output).

### Command line — interactive
```bash
python src/predict_cli.py
```
It will ask you for each value one at a time and re-ask if something's invalid (e.g. negative people, 0 counters, a bad time format).

### Web form
```bash
python src/app.py
```
Open http://127.0.0.1:5000 — fill in the form, or click one of the example preset buttons at the top to auto-fill and predict instantly.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` (or sklearn, flask, etc.) | You haven't activated the virtual environment or haven't run `pip install -r requirements.txt` yet — see Section 3. |
| `Model not found` when running `predict_cli.py` | Run the pipeline first (Option B above) so `models/queue_wait_time_model.pkl` gets created — or check you're running commands from inside the project folder, not from `src/`. |
| Path/`FileNotFoundError` errors in general | Everything must be run **from the project's root folder** (the one with this README), e.g. `python src/train.py`, **not** `cd src && python train.py` — the scripts use relative paths like `data/...` and `models/...`. |
| `pip install` fails / permission errors | Make sure the virtual environment is activated (Section 3, step 2) before installing — you should see `(venv)` in your prompt. |
| Port 5000 already in use (`app.py`) | Something else on your machine is using that port. Close it, or open `src/app.py` and change `app.run(debug=False, port=5000)` to a different port, e.g. `port=5050`. |
| Numbers look slightly different from what's already in `outputs/` | Expected if you re-ran `generate_data.py` — it uses a fixed random seed by default so results are reproducible, but if you changed `--seed` or any other argument, the dataset (and therefore the trained model's exact numbers) will differ. |

---

## 7. A note on the dataset

There's no ready-made public dataset with this project's exact combination of features (active counters, priority customers, historical load, etc.), so — as the assignment explicitly permits — this project uses a carefully designed **synthetic** dataset instead of forcing a mismatched real one into the problem. How and why it was built the way it was (including the intentional data-quality issues and the target-leakage check) is documented in `REPORT.md` and `outputs/data_quality_report.md`. If you'd rather plug in a real dataset later, `preprocess.py` expects a CSV with the same column names as `data/raw_queue_data.csv` — swap the file and re-run the pipeline from `preprocess.py` onward.
