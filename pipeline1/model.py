import pickle
from configparser import ConfigParser
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def load_model():
    config = ConfigParser()
    config.read('config.ini')

    with open(f'model/model.pkl', 'rb') as f:
        random_forest_clf = pickle.load(f)
    return random_forest_clf


def train_model(train_path, label_path):
    rfc = RandomForestClassifier()

    X_train = pd.read_csv(
        train_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    y_train = pd.read_csv(
        label_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    rfc.fit(X_train.to_numpy(), y_train['0'].to_numpy())

    with open(f'model/model.pkl', 'wb') as f:
        pickle.dump(rfc, f)
