import pickle
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_model(sequence_length):
    model = Sequential()
    model.add(InputLayer((sequence_length, 34)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(34, 'linear'))

    model.summary()
    return model


def train_model(model, train_path, val_path, sequence_length, sampling_rate):
    val = pd.read_csv(
        val_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    train = pd.read_csv(
        train_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    std_scaler = StandardScaler()
    # Fit StandardScaler only on train data, but use it to transform everything
    train_scaled = std_scaler.fit_transform(train.to_numpy())
    val_scaled = std_scaler.transform(val.to_numpy())

    X_train, y_train = x_y_from_np_array(train_scaled, train_scaled, sequence_length, sampling_rate)
    X_val, y_val = x_y_from_np_array(val_scaled, val_scaled, sequence_length, sampling_rate)

    with open('preprocessing/standard_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.002),
        metrics=[RootMeanSquaredError()]
    )

    checkpoint = ModelCheckpoint('model/', save_best_only=True, save_weights_only=True,
                                 monitor='val_root_mean_squared_error')

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=110, callbacks=[checkpoint], batch_size=60)


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
