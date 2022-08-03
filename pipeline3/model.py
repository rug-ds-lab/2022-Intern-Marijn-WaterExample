import pandas as pd
from fbprophet import Prophet
import pickle
from glob import glob
import os
from pathlib import Path


def preprocess(data):
    df = data.reset_index()
    df.set_axis(['ds', 'y'], axis=1, inplace=True)
    return df


def train_model(data_path):
    data = pd.read_csv(
        data_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    for c in data.columns:
        model = Prophet()
        train = preprocess(data[c])
        model.fit(train)
        with open(f'model/{c}.pkl', 'wb') as f:
            pickle.dump(model, f)


def load_model():
    model = {}
    for link_model_path in glob('model/*.pkl'):
        link_n = Path(link_model_path).stem
        with open(link_model_path, 'rb') as f:
            link_model = pickle.load(f)
        model[link_n] = link_model
    return model


if __name__ == '__main__':
    train_model('../dataset/leak_scenario/scenario-1/flow/train.csv')
