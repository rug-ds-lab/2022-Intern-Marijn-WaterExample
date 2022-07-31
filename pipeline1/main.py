import numpy as np
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pickle
import json
import pandas as pd
from model import load_model

KAFKA_SERVER = 'kafka:9092'


def run():
    consumer = KafkaConsumer('pressure-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline1 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    model = load_model()

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_pressure = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['1', 'time']).values
        y_pred = model.predict([current_pressure])
        y_pred_bool = np.isclose(y_pred, [1.0])[0]

        p = Point('pipeline1') \
            .field('binary_leak_prediction', bool(y_pred_bool)) \
            .time(time, write_precision='s')
        write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()
