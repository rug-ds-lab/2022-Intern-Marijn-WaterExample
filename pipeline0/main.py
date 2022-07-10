from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import pickle
import numpy as np
import pandas as pd

KAFKA_SERVER = 'kafka:9092'


def compute_correlation(v1, v2):
    correlation_matrix = np.corrcoef(v1, v2)
    return correlation_matrix


def generate_fsm_matrix(sensitivity_matrix, no_leak_signature, current_pressures):
    residual_vector = current_pressures - no_leak_signature
    fsm_matrix = \
        [compute_correlation(leak_signature.T, residual_vector) for leak_signature in sensitivity_matrix.values]
    return fsm_matrix


def run():
    consumer = KafkaConsumer('dma-epynet_data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline0 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    with open('model/sensitivity_matrix.pkl', 'rb') as f:
        sensitivity_matrix = pickle.load(f)

    with open('model/no_leak_signature.pkl', 'rb') as f:
        no_leak_signature = pickle.load(f)

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_pressure = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['1', 'time']).values
        fsm_matrix = generate_fsm_matrix(sensitivity_matrix, no_leak_signature.T, current_pressure)

        for i in range(0, len(fsm_matrix)):
            p = Point('pipeline0')\
                .tag('node', str(i+2))\
                .field('fsm_corr', fsm_matrix[i][0][1])\
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()

