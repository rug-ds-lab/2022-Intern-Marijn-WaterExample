import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model

import os
import numpy as np


def build_model():
    model = Sequential()
    model.add(InputLayer((5, 34)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(34, 'linear'))

    model.summary()
    return model


def generate_confidence_interval_bounds(y_pred, rmsfe, z_value):
    interval = z_value * rmsfe
    lower_bound = y_pred - interval
    upper_bound = y_pred + interval
    return lower_bound, upper_bound


def x_y_from_np_array(data, data_scaled, sequence_length, sampling_rate):
    X, y = [], []
    for i in range(0, len(data) - (sequence_length + 1) * sampling_rate):
        seq = [data_scaled[j] for j in range(i, i + sequence_length * sampling_rate, sampling_rate)]
        X.append(seq)
        y.append(data[i + (sequence_length + 1) * sampling_rate])
    return np.array(X), np.array(y)


def preprocess(data, sequence_length, sampling_rate, std_scaler):
    data_scaled = std_scaler.transform(data)
    X_data, y_data = x_y_from_np_array(data_scaled, data_scaled, sequence_length, sampling_rate)
    return X_data, y_data


def calculate_rmsfe(model, data_path, sequence_length, sampling_rate, std_scaler):
    val = pd.read_csv(
        data_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    X_val, y_val = preprocess(val.to_numpy(), sequence_length, sampling_rate, std_scaler)
    y_pred = model.predict(X_val)
    residuals = y_pred - y_val
    rmsfe = np.sqrt(sum([x ** 2 for x in residuals]) / len(residuals))

    return rmsfe
