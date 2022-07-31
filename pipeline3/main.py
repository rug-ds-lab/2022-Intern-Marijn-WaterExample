import os.path

import numpy as np
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pickle
import json
import pandas as pd
from datetime import datetime
from model import load_model
from configparser import ConfigParser

KAFKA_SERVER = 'kafka:9092'


def run():
    config = ConfigParser()
    config.read('config.ini')

    scenario_path = config.get('global', 'scenario_path')
    train_path = os.path.join(scenario_path, 'val.csv')

    consumer = KafkaConsumer('flow-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline1 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    model = load_model()

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_flow = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['time']).values
        time_str = datetime.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
        prediction_date = pd.DataFrame([time_str], columns=['ds'])

        y_pred = []
        upper_bound = []
        lower_bound = []

        for link_n, flow in enumerate(current_flow):
            y_pred_link = model[str(link_n+1)].predict(prediction_date)
            link_yhat = y_pred_link['yhat'][0]
            link_upper_bound = y_pred_link['yhat_upper'][0]
            link_lower_bound = y_pred_link['yhat_lower'][0]

            print(len(current_flow))
            print(current_flow)

            y_pred.append(link_yhat)
            upper_bound.append(link_upper_bound)
            lower_bound.append(link_lower_bound)

            binary_leak_prediction = flow > link_upper_bound or \
                                     flow < link_lower_bound

            p = Point('pipeline3') \
                .tag('link', str(link_n + 1)) \
                .field('flow_prediction', link_yhat) \
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)

            p = Point('pipeline3') \
                .tag('link', str(link_n + 1)) \
                .field('flow_prediction_upper_bound', link_upper_bound) \
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)

            p = Point('pipeline3') \
                .tag('link', str(link_n + 1)) \
                .field('flow_prediction_lower_bound', link_lower_bound) \
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)

            p = Point('pipeline3') \
                .field('binary_leak_prediction', bool(binary_leak_prediction)) \
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()