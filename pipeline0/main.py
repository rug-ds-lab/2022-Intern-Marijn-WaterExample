from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import pandas as pd
from configparser import ConfigParser
from model import train_model, load_model, generate_fsm_matrix, generate_binary_leak_prediction

KAFKA_SERVER = 'kafka:9092'


def run():
    config = ConfigParser()
    config.read('config.ini')
    water_metric = config.get('pipeline0', 'water_metric')

    correlation_threshold = config.getfloat('pipeline0', 'correlation_threshold')
    retrain_model = config.getboolean('pipeline0', 'train_model')

    if retrain_model:
        train_model()

    sensitivity_matrix, no_leak_signature = load_model()
    consumer = KafkaConsumer(f'{water_metric}-data', bootstrap_servers=KAFKA_SERVER)

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for msg in consumer:
        time = int(json.loads(msg.value.decode('utf-8'))['time'])
        current_pressure = pd.Series(json.loads(msg.value.decode('utf-8'))).drop(labels=['time']).values
        fsm_matrix = generate_fsm_matrix(sensitivity_matrix, no_leak_signature.T, current_pressure)
        binary_leak_prediction = generate_binary_leak_prediction(fsm_matrix, correlation_threshold)

        p = Point('pipeline0')\
            .field('binary_leak_prediction', binary_leak_prediction)\
            .time(time, write_precision='s')
        write_api.write(bucket='primary', record=p)

        for i in range(0, len(fsm_matrix)):
            p = Point('pipeline0')\
                .tag('node', str(i+2))\
                .field('fsm_corr', fsm_matrix[i][0][1])\
                .time(time, write_precision='s')
            write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()

