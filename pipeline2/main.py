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

KAFKA_SERVER = 'kafka:9092'


def run():
    consumer = KafkaConsumer('flow-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline2 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    # tables = query_api('')

    with open('preprocessing/standard_scaler.pkl', 'rb') as f:
        # This standard scaler is fitted to training data only
        standard_scaler = pickle.load(f)

    with open('postprocessing/rmsfe_vector.pkl', 'rb') as f:
        # The RMSFE values here are calculated based on validation set performance and are used to calculate the CI
        rmsfe_vector = pickle.load(f)

    model = load_model('model')

    last_n_values = []
    sequence_length = 5

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_flow = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['time']).values

        last_n_values.append(current_flow)

        if len(last_n_values) > sequence_length:
            last_n_values.pop(0)

        if len(last_n_values) == sequence_length:
            last_n_values_scaled = standard_scaler.transform(last_n_values)
            y_pred = model.predict(np.array([last_n_values_scaled]))
            print(last_n_values_scaled)

            y_pred_unscaled = standard_scaler.inverse_transform(y_pred)
            y_pred_upper_threshold = standard_scaler.inverse_transform([y_pred[0] + 1.96 * rmsfe_vector])
            y_pred_lower_threshold = standard_scaler.inverse_transform([y_pred[0] - 1.96 * rmsfe_vector])

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
