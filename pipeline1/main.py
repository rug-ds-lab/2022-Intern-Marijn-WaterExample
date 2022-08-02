import numpy as np
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pickle
import json
import pandas as pd
from model import load_model, train_model
from configparser import ConfigParser
import os

KAFKA_SERVER = 'kafka:9092'


def run():
    config = ConfigParser()
    config.read('config.ini')

    water_metric = config.get('pipeline1', 'water_metric')
    retrain_model = config.getboolean('pipeline1', 'train_model')
    scenario_path = config.get('global', 'scenario_path')
    train_scenario_path = config.get('pipeline1', 'train_scenario_path')

    # Train on test set of other scenario here since we need a leak scenario to train the RFC model
    train_path = os.path.join(train_scenario_path, water_metric, 'test.csv')
    label_path = os.path.join(scenario_path, 'labels.csv')

    consumer = KafkaConsumer(f'{water_metric}-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline1: Started')
    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    if retrain_model:
        train_model(train_path, label_path)

    model = load_model()

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_pressure = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['time']).values

        y_pred = model.predict([current_pressure])
        y_pred_bool = np.isclose(y_pred, [1.0])[0]

        p = Point('pipeline1') \
            .field('binary_leak_prediction', bool(y_pred_bool)) \
            .time(time, write_precision='s')
        write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()
