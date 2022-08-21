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
    """
    Defines the architecture of the Keras model

    :param sequence_length:
    :return: keras model
    """


    model = Sequential()
    model.add(InputLayer((sequence_length, 34)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(34, 'linear'))

    model.summary()
    return model


def train_model(model, train_path, val_path, sequence_length, sampling_rate):
    """
    Splits a training set into small series suitable for a training a time series model, then trains a keras model on
    that training set

    :param model:
    :param train_path:
    :param val_path:
    :param sequence_length: look back for the time-series model
    :param sampling_rate: how many values to skip ahead each time when constructing series for training
    :return:
    """

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

    # Save the model with the best performance on the validation set throughout all epochs
    checkpoint = ModelCheckpoint('model/', save_best_only=True, save_weights_only=True,
                                 monitor='val_root_mean_squared_error')

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=110, callbacks=[checkpoint], batch_size=60)


def generate_confidence_interval_bounds(y_pred, rmsfe, z_value):
    """
    Generates bounds of a confidence interval based on a given root-mean-square forecasting error (rmsfe) value and a
    z value.

    :param y_pred: predicted values
    :param rmsfe: rmsfe value which describes the performance of the model (usually on the validation set)
    :param z_value: z value describing the desired confidence
    :return:
    """

    interval = z_value * rmsfe
    lower_bound = y_pred - interval
    upper_bound = y_pred + interval
    return lower_bound, upper_bound


def x_y_from_np_array(data, data_scaled, sequence_length, sampling_rate):
    """
    This generates small series based on a list of time-series data. This is done to make the data suitable for training
    a time-series model. The series are of the length (sequence_length) and have single label which is the next value in
    the series.

    :param data:
    :param data_scaled: the data in scaled format
    :param sequence_length: length of the generated series
    :param sampling_rate: how many values to skip ahead in the original series when generating the new set of series
    :return:
    """


    X, y = [], []
    for i in range(0, len(data) - (sequence_length + 1) * sampling_rate):
        seq = [data_scaled[j] for j in range(i, i + sequence_length * sampling_rate, sampling_rate)]
        X.append(seq)
        y.append(data[i + (sequence_length + 1) * sampling_rate])
    return np.array(X), np.array(y)


def preprocess(data, sequence_length, sampling_rate, std_scaler):
    """
    This function applies all preprocessing functions to the data

    :param data:
    :param sequence_length: sequence length for x_y_from_np_array
    :param sampling_rate: sampling rate for x_y_from_np_array
    :param std_scaler: scaler file that was fit to the training data
    :return:
    """

    data_scaled = std_scaler.transform(data)
    X_data, y_data = x_y_from_np_array(data_scaled, data_scaled, sequence_length, sampling_rate)
    return X_data, y_data


def calculate_rmsfe(model, data_path, sequence_length, sampling_rate, std_scaler):
    """
    Calculate the root-mean-square forecasting error based on the performance on the validation set

    :param model:
    :param data_path:
    :param sequence_length:
    :param sampling_rate:
    :param std_scaler:
    :return:
    """


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
