from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pickle
import json
import pandas as pd

KAFKA_SERVER = 'kafka:9092'


def run():
    consumer = KafkaConsumer('pressure-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline1 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    with open('model/random_forest_clf_n=5000_s21-300.pkl', 'rb') as f:
        random_forest_clf = pickle.load(f)

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_pressure = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['1', 'time']).values

        y_pred = random_forest_clf.predict([current_pressure])

        p = Point('pipeline1') \
            .field('leak_detection', y_pred[0]) \
            .time(time, write_precision='s')
        write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()
