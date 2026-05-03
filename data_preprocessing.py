import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
    # Map target labels (optional)
        df["target"] = df["species"].map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
        })
        drop_columns = ['Id', 'species']
        df = df.drop(columns=drop_columns, errors='ignore')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

    
#final_df = preprocess_data(df)

def main():
    try:
        train_data = pd.read_csv(r'./data/raw/train.csv')
        test_data = pd.read_csv(r'./data/raw/test.csv')
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)
        save_data(train_data, test_data, data_path='./data/final')
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
        
#train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
if __name__ == "__main__":
    main()