#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import os
import logging
#from data_ingestion import load_data


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
def save_model(model, model_name: str, save_dir: str = r"D:\Projects\models") -> str:
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")
    return model_path


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
    train_df = pd.read_csv(r"D:\Projects\src\data\final\raw\train_final.csv")
    test_df = pd.read_csv(r"D:\Projects\src\data\final\raw\test_final.csv")
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    try:
        models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            LogisticRegression(max_iter=200),
            DecisionTreeClassifier(random_state=42),
            SVC()
        ]

        for model in models:
            result = train_and_evaluate(model, X_train, X_test, y_train, y_test)
            print(f"{result['model']}: {result['accuracy']:.4f}")

        results = compare_models(models, X_train, X_test, y_train, y_test)
        for r in results:
            print(f"{r['model']}: {r['accuracy']:.4f}")

        # Save each trained model so model_evalution.py can load them
        trained_models = {m.__class__.__name__: m for m in models}
        for name, model in trained_models.items():
            save_model(model, name)

        logger.info("Model training, evaluation and saving completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main train and testing model: {e}")
        raise