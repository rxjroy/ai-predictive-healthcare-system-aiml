Summary
An AI-powered predictive healthcare system that ingests structured clinical data, preprocesses features, trains multiple classifiers (RF, LR, SVM, KNN, Naïve Bayes), and serves risk predictions via a reproducible Colab pipeline and a lightweight Flask API for inference.

Features
End-to-end ML pipeline: data cleaning, encoding, normalization, feature selection, model training, evaluation, and artifact export.

Comparative modeling: Random Forest, Logistic Regression, SVM, KNN, and Naïve Bayes with k-fold CV and hyperparameter tuning.

Evaluation suite: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification report heatmaps.

Deployment-ready: packaged preprocessing + model with a Flask API for risk prediction endpoints.

Tech stack
Python 3.x, Google Colab for experimentation and Drive integration.

Libraries: scikit-learn, pandas, numpy, matplotlib; optional TensorFlow for extensions.

Web: Flask microservice for inference and simple input validation.

Dataset
Schema: 2,219 rows × 9 attributes—Condition, Drug, Indication, Type, Reviews, Effective, EaseOfUse, Satisfaction, Information.

Notes: numeric ratings are normalized; categorical attributes are encoded; unstructured Information text reserved for future NLP features.

Ethics: uses public data; remove PHI; follow dataset license and responsible AI guidelines.

Project structure
notebooks/

01_data_preprocessing.ipynb — cleaning, encoding, normalization, splits.

02_model_training.ipynb — RF, LR, SVM, KNN, NB with CV + tuning.

03_evaluation_reporting.ipynb — confusion matrices, ROC curves, heatmaps.

src/

data_pipeline.py — sklearn Pipeline for preprocessing and feature selection.

models.py — constructors for RF/LR/SVM/KNN/NB with standard fit/predict.

train.py — training, CV, hyperparameter search, artifact saving.

evaluate.py — metrics computation and plots.

api.py — Flask app loading persisted pipeline + model.

artifacts/

models/ — persisted joblib models and preprocessing objects.

metrics/ — CSV/JSON reports and CV logs.

plots/ — confusion matrices, ROC, heatmaps.

configs/

experiment.yaml — seeds, CV folds, scoring, model hyperparameters.

Setup
Colab

Open notebooks in Google Colab, mount Drive, set DATA_PATH and OUTPUT_DIR.

Install deps in first cell: scikit-learn, pandas, numpy, matplotlib, flask.

Run notebooks in order: 01 → 02 → 03 to regenerate artifacts.

Local

Python ≥ 3.10 recommended.

pip install -r requirements.txt.

Place raw data at data/raw/dataset.csv and update configs/experiment.yaml.

Quick start
Train

python -m src.train --data data/processed/train.csv --model rf --config configs/experiment.yaml --out artifacts

Evaluate

python -m src.evaluate --model artifacts/models/rf.pkl --test data/processed/test.csv --report artifacts/metrics/rf_report.json

Serve API

python -m src.api # starts Flask and loads preprocessing + model bundle

Modeling
Algorithms: Random Forest, Logistic Regression, SVM (linear/RBF), KNN, Naïve Bayes.

Tuning: grid/random search over RF depth/trees, LR C/penalty, SVM C/gamma, KNN neighbors/metric; NB minimal tuning.

Protocol: stratified train/test split (70–80/20–30), k-fold CV for selection, held-out test for final reporting.

Evaluation
Metrics: accuracy, precision, recall, macro F1; one-vs-rest ROC-AUC for multiclass; confusion matrices and classification report heatmaps.

Artifacts: CSV logs, per-model plots in artifacts/plots for reproducibility and comparison.

API usage
Endpoint: POST /predict

Input JSON example:

{ "Condition": "...", "Drug": "...", "Indication": "On Label", "Type": "RX", "Effective": 8, "EaseOfUse": 7, "Satisfaction": 8 }

Output example:

{ "prediction": "disease_label", "confidence": 0.84, "top_classes": [{"label":"A","p":0.84},{"label":"B","p":0.12}] }

Results snapshot
The report includes confusion matrices, classification report heatmaps, and ROC curves for KNN, SVM, LR, and Naïve Bayes, alongside a performance matrix table. Replace this section with exact numbers after running evaluation.

Guidance: prefer the most stable model across folds balancing macro F1 and interpretability; RF and LR often offer strong baselines with explainability.

Explainability
Global: feature importance for RF; coefficients and odds ratios for LR.

Roadmap: add SHAP summaries for local and global explanations as an extension.

Limitations
No clinical deployment; no real-time data capture; classical ML only (no deep learning in-scope).

Potential class imbalance and dataset-specific generalization limits; requires domain validation and calibration.

Roadmap
Add PR-AUC and calibration curves; SHAP explainability; Optuna/MLflow integration; Dockerization; ONNX export; basic UI.

How to cite
If using this repository, cite the project and the dataset provider per their licenses; include a BibTeX entry in docs/citation.bib.

Contributors
Raj Roy, Saksham Kumar Rana, Gopal Gohel — Marwadi University, Department of Computer Engineering.
