# Iris Classification — MLOps Pipeline

End-to-end machine learning pipeline for Iris species classification, built with DVC for pipeline orchestration and experiment reproducibility.

## Project Structure

```
src/
├── data/
│   ├── raw/                  # Ingested train/test splits
│   └── final/                # Preprocessed train/test splits
├── models/
│   └── model.pkl             # Best trained model (DVC-tracked)
├── reports/
│   └── evaluation.txt        # Evaluation metrics
├── logs/                     # Per-stage log files
├── data_ingestion.py
├── data_preprocessing.py
├── model_training.py
├── model_evalution.py
└── dvc.yaml                  # Pipeline definition
```

## Pipeline Stages

```
data_ingestion → data_preprocessing → model_training → model_evaluation
```

| Stage | Script | Input | Output |
|---|---|---|---|
| `data_ingestion` | `data_ingestion.py` | UCI Iris dataset (URL) | `data/raw/train.csv`, `data/raw/test.csv` |
| `data_preprocessing` | `data_preprocessing.py` | `data/raw/*.csv` | `data/final/train.csv`, `data/final/test.csv` |
| `model_training` | `model_training.py` | `data/final/train.csv` | `models/model.pkl` |
| `model_evaluation` | `model_evalution.py` | `models/model.pkl`, `data/final/test.csv` | `reports/evaluation.txt` |

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/mdzeeshan31/Mlops.git
cd Mlops
```

**2. Create and activate a virtual environment**
```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate
```

**3. Install dependencies**
```bash
pip install dvc scikit-learn pandas joblib pyyaml
```

## Running the Pipeline

Run all stages:
```bash
dvc repro
```

Run a specific stage:
```bash
dvc repro model_training
```

DVC skips stages whose inputs have not changed, so re-running is fast.

## Results

Best model: **RandomForestClassifier**

| Metric | Score |
|---|---|
| Accuracy | 96.67% |
| Macro F1 | 0.96 |

Full report is written to `reports/evaluation.txt` after `dvc repro`.

## Models Compared

The training stage evaluates four classifiers and saves the best-performing one:

- Random Forest
- Logistic Regression
- Decision Tree
- SVM (SVC)

## Logging

Each stage writes structured logs to `logs/`:

- `logs/data_ingestion.log`
- `logs/data_preprocessing.log`
- `logs/model_training.log`
- `logs/model_evaluation.log`
