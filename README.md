# Heart Disease Risk

A dark-mode Streamlit app that estimates a patient's risk of heart disease from 11 clinical readings. It runs two models trained on the UCI Heart Disease data and updates the result live as you change any input.

**Live app:** https://heart-disease-prediction-mlp-lr.streamlit.app/ *(replace after deploying)*

> **Not a medical device.** This project is for exploration and learning. It must not replace a clinician's judgment.

## What it does

- Takes 11 inputs: age, sex, chest pain type, exercise-induced angina, resting blood pressure, cholesterol, max heart rate, fasting blood sugar, resting ECG, ST depression, and thallium stress result.
- Shows the predicted probability of heart disease and a risk band: **Low**, **Moderate** or **High**.
- Lets you switch between a logistic regression model and a neural network, and shows both scores side by side.
- Includes an **About the model** page that explains how the models were trained and how well they perform.

### Risk bands

The models were trained on two classes (disease or no disease), so *Moderate* is not a trained label. It is a band of uncertainty on the predicted probability.

| Probability | Band |
|---|---|
| Under 35% | Low |
| 35% to 65% | Moderate |
| 65% or more | High |

The 35% cutoff comes from the threshold analysis in the notebook, and 65% mirrors it. To change them, edit `LOW_CUTOFF` and `HIGH_CUTOFF` at the top of `app.py`. The probabilities are not calibrated.

## The models

| Model | Test ROC-AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Logistic regression (L2, C = 10), cutoff 0.50 | 0.931 | 85% | 93% | 76% |
| Neural network (18 → 100 tanh → 1), cutoff 0.50 | not measured | 85% | 93% | 76% |

- **Data:** 918 patients from four hospitals (Cleveland, Hungary, Switzerland, VA Long Beach). Severity labels 1 to 4 were merged into a single "disease" class.
- **Split:** 80% train, 20% test, stratified, seed 42 (734 and 184 patients).
- **Features:** 18 after one-hot encoding. `ca` and `slope` were dropped because too many values were missing.
- **Logistic regression:** grid search over regularization strength and L1/L2 penalty with 5-fold cross-validation, scored on ROC-AUC.
- **Neural network:** written from scratch in PyTorch and trained with binary cross-entropy and full-batch gradient descent (learning rate 0.01, 2,100 steps).

The full workflow, including exploratory analysis, imputation and a random forest comparison, is in [`Heart_Disease_Prediction.ipynb`](Heart_Disease_Prediction.ipynb).

## Repository layout

```
app.py                         Streamlit app (both pages)
requirements.txt               Python dependencies
.streamlit/config.toml         Dark theme
scaler.pkl                     Fitted StandardScaler
log_reg_model.pkl              Trained logistic regression
mlp_weights.pt                 Trained neural network weights
expected_columns.pkl           Feature order used in training
Heart_Disease_Prediction.ipynb Training notebook
```

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

`scikit-learn` and `torch` should match the versions you trained with, because the saved model files are version-sensitive. If you see an unpickling error, run `pip freeze` in your training environment and pin those versions in `requirements.txt`.

## Deploy

GitHub Pages only serves static files, so it can't run this app. Deploy it on [Streamlit Community Cloud](https://share.streamlit.io) instead, which is free and builds straight from this repository:

1. Make sure `requirements.txt`, `app.py`, the model files and `.streamlit/config.toml` are committed.
2. Sign in at https://share.streamlit.io with your GitHub account.
3. Select **Create app**, then choose this repository, the branch, and `app.py` as the main file.
4. Select **Deploy** and wait for the build to finish.

## Limitations

- The dataset is small (918 patients) and has not been checked on any outside population.
- The imputers were fit on all rows before the train/test split, so test scores may be slightly optimistic.
- The test set has 184 patients, so the metrics can shift by several points with a different split.

## Data source

UCI Heart Disease dataset (Janosi, Steinbrunn, Pfisterer and Detrano), loaded in the notebook from the Kaggle copy `redwankarimsony/heart-disease-data`.
