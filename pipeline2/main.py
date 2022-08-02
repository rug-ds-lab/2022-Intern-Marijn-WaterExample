import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model

from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from model import build_model, generate_confidence_interval_bounds, calculate_rmsfe, train_model
from configparser import ConfigParser
import os

KAFKA_SERVER = 'kafka:9092'


def run():
    config = ConfigParser()
    config.read('config.ini')

    z_value = config.getfloat('pipeline2', 'z_value')
    scenario_path = config.get('global', 'scenario_path')
    sequence_length = config.getint('pipeline2', 'sequence_length')
    sampling_rate = config.getint('pipeline2', 'sampling_rate')
    water_metric = config.get('pipeline2', 'water_metric')

    consumer = KafkaConsumer(f'{water_metric}-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline2 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    retrain_model = config.get('pipeline2', 'train_model')

    train_path = os.path.join(scenario_path, water_metric, 'train.csv')
    val_path = os.path.join(scenario_path, water_metric, 'val.csv')

    model = build_model(sequence_length)

    if retrain_model:
        train_model(model, train_path, val_path, sequence_length, sampling_rate)

    with open('preprocessing/standard_scaler.pkl', 'rb') as f:
        # This standard scaler is fitted to training data only
        standard_scaler = pickle.load(f)

    model.load_weights('model/')

    val_path = os.path.join(scenario_path, 'flow', 'val.csv')

    rmsfe = calculate_rmsfe(
        model=model,
        data_path=val_path,
        sequence_length=sequence_length,
        sampling_rate=sampling_rate,
        std_scaler=standard_scaler
    )

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_flow = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['time']).values

        query = f' from(bucket:"primary") ' \
                f'|> range(start: {time-sequence_length*24*60*60}, stop: {time})' \
                f'|> filter(fn: (r) => r._field == "flow")' \
                f'|> group(columns: ["link"], mode: "by")'

        result = query_api.query_data_frame(org='primary', query=query)
        flow_as_matrix = result.pivot(index='_time', columns='link', values='_value')
        flow_as_matrix.columns = flow_as_matrix.columns.astype('int64')
        flow_as_matrix_sorted = flow_as_matrix.reindex(columns=flow_as_matrix.columns.sort_values())

        print(flow_as_matrix_sorted)

        if flow_as_matrix.shape[0] == sequence_length * sampling_rate:
            last_n_values = flow_as_matrix_sorted.loc[
                (flow_as_matrix.index.hour == datetime.fromtimestamp(time).hour)
                & (flow_as_matrix.index.minute == datetime.fromtimestamp(time).minute)]

            print(f'Current time: {datetime.fromtimestamp(time)}')
            print(last_n_values)

            last_n_values_scaled = standard_scaler.transform(last_n_values)
            y_pred = model.predict(np.array([last_n_values_scaled]))
            y_pred_unscaled = standard_scaler.inverse_transform(y_pred)

            lower_threshold, upper_threshold = generate_confidence_interval_bounds(y_pred[0], rmsfe, z_value)
            lower_threshold_unscaled = standard_scaler.inverse_transform([lower_threshold])
            upper_threshold_unscaled = standard_scaler.inverse_transform([upper_threshold])

            binary_leak_prediction = any(current_flow[i] > upper_threshold_unscaled[0][i] or
                                         y_pred_unscaled[0][i] < lower_threshold_unscaled[0][i]
                                         for i in range(0, len(y_pred_unscaled[0])))

            p = Point('pipeline2') \
                .field('binary_leak_prediction', binary_leak_prediction) \
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)

            print(y_pred)

            for link_n, flow_value in enumerate(y_pred_unscaled[0]):
                p = Point('pipeline2') \
                    .tag('link', str(link_n + 1)) \
                    .field('flow_prediction', flow_value) \
                    .time(time, write_precision='s')
                write_api.write(bucket='primary', record=p)

                p = Point('pipeline2') \
                    .tag('link', str(link_n + 1)) \
                    .field('flow_prediction_lower_threshold', lower_threshold_unscaled[0][link_n]) \
                    .time(time, write_precision='s')
                write_api.write(bucket='primary', record=p)

                p = Point('pipeline2') \
                    .tag('link', str(link_n + 1)) \
                    .field('flow_prediction_upper_threshold', upper_threshold_unscaled[0][link_n]) \
                    .time(time, write_precision='s')
                write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()
