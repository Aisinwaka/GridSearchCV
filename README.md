# Capstone: Advanced Predictive Modeling for Airline Satisfaction — XGBoost Deployment

## Dataset

`airline_satisfaction.csv` — 4,000 passenger records with demographics (Gender, Age,
Customer Type), trip details (Type of Travel, Class, Flight Distance), 14 service-quality
ratings (0–5 scale, e.g. wifi, seat comfort, online boarding, cleanliness), delay minutes,
and the binary target `satisfaction` (`satisfied` vs. `neutral or dissatisfied`).

*Note: this is a synthetic dataset generated to mirror the structure and realistic
relationships of the standard airline passenger satisfaction survey schema, used in place
of the original course dataset for this capstone.*

## Discovery-to-Action (DTA) Workflow

**1. Discovery Phase**
- Loaded and inspected the dataset; imputed the small number of missing
  `Arrival Delay in Minutes` values with the median (robust to the right-skewed delay
  distribution).
- Label-encoded binary categoricals (`Gender`, `Customer Type`, `Type of Travel`),
  ordinally mapped `Class` (Eco < Eco Plus < Business), and encoded the target.
- Split into an 80/20 train/test set (stratified) and reused this **identical split and
  feature set** across all three models for a fair, apples-to-apples comparison.

**2. Technical Phase (Gradient Boosting)**
- Fit Decision Tree and Random Forest baselines on the shared pipeline.
- Implemented `XGBClassifier` and tuned `max_depth` × `learning_rate` via 5-fold
  `GridSearchCV`, optimizing for F1-score.
- **Best parameters:** `learning_rate=0.1`, `max_depth=3`, `n_estimators=150`
  (best CV F1 ≈ 0.824).
- Trained the final tuned model and generated test-set predictions.

**Model comparison (test set):**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree (prior project) | 0.739 | 0.784 | 0.737 | 0.760 |
| Random Forest (prior project) | 0.804 | 0.833 | 0.813 | 0.823 |
| **XGBoost (current implementation)** | **0.808** | **0.833** | **0.821** | **0.827** |

**3. Action Phase**
- **Recommended model: XGBoost** — highest F1-score and recall on the satisfied class,
  which matters because a false negative (missing a dissatisfied customer) is more costly
  to the airline than a false positive.
- **Feature importance:** `Class` and `Type of Travel` are the dominant predictors, followed
  by `Flight Distance` and both delay features — trip context and disruption matter more
  than any single service rating. Among service-quality features specifically, **seat
  comfort** and **check-in efficiency** carry the most weight.

## Business Recommendation

- Prioritize **delay reduction** and **class-tailored service standards** — the two biggest
  levers in the model.
- Within service quality, invest specifically in **seat comfort** and **check-in
  efficiency**, the highest-impact individual touchpoints.
- Use the model to proactively flag at-risk (predicted-dissatisfied) passengers for
  targeted service recovery before they churn.

## Limitations & Deployment Notes

- Built on a synthetic dataset; validate on live production data before full rollout.
- Service ratings are self-reported and carry inherent subjective bias.
- Recommended rollout: pilot on a single route/hub, monitor precision/recall monthly,
  retrain quarterly (or on detected performance drift), and version the model artifact
  together with its encoding maps to avoid silent inference breakage.

## Setup Instructions

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook capstone_xgboost_airline.ipynb
```

## Files

- `capstone_xgboost_airline.ipynb` — full analysis notebook (all cells executed: GridSearchCV
  output, model comparison table, confusion matrices, feature importance plot, executive
  summary)
- `airline_satisfaction.csv` — dataset
- `README.md` — this file
