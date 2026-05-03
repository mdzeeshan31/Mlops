from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import joblib
import os
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

MODELS_DIR = r"D:\Projects\models"


def load_model(model_name: str, models_dir: str = MODELS_DIR):
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run model_training.py first.")
    model = joblib.load(model_path)
    logger.info(f"Loaded model: {model_path}")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logger.info(f"{model.__class__.__name__} accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating {model.__class__.__name__}: {e}")
        raise

    return {
        "model": model.__class__.__name__,
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": y_pred
    }


def main():
    test_df = pd.read_csv(r"D:\Projects\src\data\final\raw\test_final.csv")
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    model_names = [
        "RandomForestClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "SVC",
    ]

    results = []
    for name in model_names:
        try:
            model = load_model(name)
            result = evaluate_model(model, X_test, y_test)
            results.append(result)
            print(f"\n{result['model']} — Accuracy: {result['accuracy']:.4f}")
            print(result['classification_report'])
        except FileNotFoundError as e:
            logger.warning(str(e))

    if results:
        best = max(results, key=lambda r: r["accuracy"])
        print(f"\nBest model: {best['model']} ({best['accuracy']:.4f})")
        logger.info(f"Evaluation complete. Best model: {best['model']} ({best['accuracy']:.4f})")


if __name__ == "__main__":
    main()
