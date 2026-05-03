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

def load_model(model_path: str):
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
    test_df = pd.read_csv('./data/final/test.csv')
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    model = load_model('./models/model.pkl')
    result = evaluate_model(model, X_test, y_test)

    output = (
        f"Model: {result['model']}\n"
        f"Accuracy: {result['accuracy']:.4f}\n\n"
        f"{result['classification_report']}"
    )
    print(output)

    os.makedirs('reports', exist_ok=True)
    with open('reports/evaluation.txt', 'w') as f:
        f.write(output)
    logger.info(f"Evaluation complete. Accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
