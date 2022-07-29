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

KAFKA_SERVER = 'kafka:9092'


def run():
    consumer = KafkaConsumer('flow-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline2 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    with open('preprocessing/standard_scaler.pkl', 'rb') as f:
        # This standard scaler is fitted to training data only
        standard_scaler = pickle.load(f)

    with open('postprocessing/rmsfe_vector.pkl', 'rb') as f:
        # The RMSFE values here are calculated based on validation set performance and are used to calculate the CI
        rmsfe_vector = pickle.load(f)

    model = Sequential()
    model.add(InputLayer((5, 34)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(34, 'linear'))

    model.summary()

    model.load_weights('model/')

    sequence_length = 5

    # This is the amount of time steps before we are at the same time of the day again
    time_interval = 48

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

        if flow_as_matrix.shape[0] == sequence_length * time_interval:
            last_n_values = flow_as_matrix_sorted.loc[
                (flow_as_matrix.index.hour == datetime.fromtimestamp(time).hour)
                & (flow_as_matrix.index.minute == datetime.fromtimestamp(time).minute)]

            print(f'Current time: {datetime.fromtimestamp(time)}')
            print(last_n_values)

            last_n_values_scaled = standard_scaler.transform(last_n_values)
            y_pred = model.predict(np.array([last_n_values_scaled]))

            y_pred_unscaled = standard_scaler.inverse_transform(y_pred)
            y_pred_upper_threshold = standard_scaler.inverse_transform([y_pred[0] + 1.96 * rmsfe_vector])
            y_pred_lower_threshold = standard_scaler.inverse_transform([y_pred[0] - 1.96 * rmsfe_vector])

            print("Y_PRED: ")
            print(y_pred)

            for link_n, flow_value in enumerate(y_pred_unscaled[0]):
                p = Point('pipeline2') \
                    .tag('link', str(link_n + 1)) \
                    .field('flow_prediction', flow_value) \
                    .time(time, write_precision='s')
                write_api.write(bucket='primary', record=p)

                p = Point('pipeline2') \
                    .tag('link', str(link_n + 1)) \
                    .field('flow_prediction_upper_threshold', y_pred_upper_threshold[0][link_n]) \
                    .time(time, write_precision='s')
                write_api.write(bucket='primary', record=p)

                p = Point('pipeline2') \
                    .tag('link', str(link_n + 1)) \
                    .field('flow_prediction_lower_threshold', y_pred_lower_threshold[0][link_n]) \
                    .time(time, write_precision='s')
                write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()
