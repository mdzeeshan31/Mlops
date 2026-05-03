from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import logging


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#X_train, X_test, y_train, y_test = train_test_split(, y, test_size=0.2, random_state=42)

# Generic function for training + evaluation
def save_model(model, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")


def train_and_evaluate(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train any sklearn model and return accuracy
    """
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error occurred while training {model.__class__.__name__}: {e}")
        raise

    return {
        "model": model.__class__.__name__,
        "accuracy": accuracy,
        "predictions": y_pred
    }



def compare_models(models, X_train, X_test, y_train, y_test):
    results = []

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"{model.__class__.__name__}: {accuracy:.4f}")
        logger.debug("test_prediction is sufficiently accurate:")
    
        results.append({
            "model": model.__class__.__name__,
            "accuracy": accuracy
        })

    return sorted(results, key=lambda x: x["accuracy"], reverse=True)


def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['model_training']

    train_df = pd.read_csv('./data/final/train.csv')
    test_df = pd.read_csv('./data/final/test.csv')
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    try:
        models = [
            RandomForestClassifier(**params['random_forest']),
            LogisticRegression(**params['logistic_regression']),
            DecisionTreeClassifier(**params['decision_tree']),
            SVC()
        ]

        results = compare_models(models, X_train, X_test, y_train, y_test)
        for r in results:
            print(f"{r['model']}: {r['accuracy']:.4f}")

        best_model_name = results[0]['model']
        best_model = next(m for m in models if m.__class__.__name__ == best_model_name)
        save_model(best_model, './models/model.pkl')

        logger.info("Model training, evaluation and saving completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main train and testing model: {e}")
        raise

if __name__ == "__main__":
    main()