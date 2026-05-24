# Turbofan Predictive Maintenance

Predicting **Remaining Useful Life (RUL)** of aircraft turbofan engines and scheduling
their maintenance, by combining gradient-boosted regression with evolutionary
optimization. Built on NASA's C-MAPSS run-to-failure dataset.

---

## Overview

Unplanned engine failures are expensive and dangerous; replacing healthy engines too
early is wasteful. This project tackles both sides of that trade-off in two stages:

1. **Prediction** – an XGBoost regressor estimates the RUL (in cycles) of in-service
   engines from their sensor degradation patterns.
2. **Prescription** – a Genetic Algorithm (GA) uses those RUL estimates to build a
   maintenance schedule that minimizes total cost, balancing the risk of failure
   against the cost of premature servicing.

This was originally developed as a Prescriptive Analytics assignment (JM0100) and has
been cleaned up here as a portfolio project.

---

## Dataset

[NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

- `DataTrain.csv` – complete run-to-failure trajectories used to train and validate the
  RUL model.
- `DataSchedule.csv` – in-service engines (truncated trajectories) for which a
  maintenance schedule is produced.

Each row is one operational cycle of one engine, with 21 sensor readings and 3
operational-setting columns.

---

## Methodology

### RUL prediction (XGBoost)

- **RUL clipping at 125 cycles.** Early in an engine's life, sensor signals are
  indistinguishable from baseline noise, so RUL is capped at 125. This is the
  defensible choice in the literature (Heimes, 2008; Zheng et al., 2017): it preserves
  dynamic range while avoiding meaningless targets for healthy engines.
- **Engine-level splitting.** Train/test splits and cross-validation are done *per
  engine* using `GroupKFold` to prevent data leakage — cycles from the same engine
  never appear in both train and test.
- **Feature selection.** The raw `cycle` counter is deliberately dropped so the model
  cannot use elapsed time as a shortcut for RUL; predictions are driven by sensor
  degradation instead.
- **Hyperparameter tuning** via `RandomizedSearchCV` (Bergstra & Bengio, 2012).

### Maintenance scheduling (Genetic Algorithm)

- Implemented with the **DEAP** evolutionary-computation framework.
- A two-tier mutation structure: an individual-level mutation probability
  (`mutation_prob`) decides *whether* an individual mutates, and a gene-level
  probability (`gene_mutation_prob`) decides *which* genes change.
- The fitness function trades off failure penalties against maintenance cost to find a
  low-cost feasible schedule.

---

## Results

- The XGBoost model recovers engine degradation well across the test engines (see the
  report for MAE / RMSE and the predicted-vs-actual plots).
- A benchmark against an external set of predictions showed a strong correlation
  (r ≈ 0.91) but a statistically significant **negative bias** of a few cycles — a nice
  illustration of how a single preprocessing choice (RUL clipping) propagates all the
  way into downstream outputs.
- The GA produces a maintenance schedule that reduces total expected cost relative to a
  naive fixed-interval policy.

> Final metrics, tables, and figures are in [`report/report.pdf`](report/report.pdf).



## Getting started

```bash
# 1. Clone
git clone https://github.com/<your-username>/turbofan-predictive-maintenance.git
cd turbofan-predictive-maintenance

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/turbofan_rul_maintenance.ipynb
```

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · DEAP · matplotlib · seaborn


Developed for the JM0100 Prescriptive Analytics course.
