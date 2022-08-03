import time

import os
import pandas as pd
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from configparser import ConfigParser
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score

KAFKA_SERVER = 'kafka:9092'


def compute_performance_metric(y_true, y_pred, metric):
    performance = metric(y_true.to_numpy(), y_pred.to_numpy())
    print(y_true)
    print(y_pred)
    print(classification_report(y_true, y_pred))
    return performance


def run():
    config = ConfigParser()
    config.read('config.ini')

    scenario_path = config.get('global', 'scenario_path')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    label_path = os.path.join(scenario_path, 'labels.csv')

    labels = pd.read_csv(
        label_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    while True:
        time.sleep(2)
        prediction_query = f' from(bucket:"primary") ' \
                           f'|> range(start: 2017-01-01T00:00:00Z, stop: 2017-12-31T12:00:00Z)' \
                           f'|> filter(fn: (r) => r._field == "binary_leak_prediction")' \
                           f'|> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")'
        result = query_api.query_data_frame(org='primary', query=prediction_query)
        print(result.columns)

        ground_truth_query = f' from(bucket:"primary") ' \
                             f'|> range(start: 2017-01-01T00:00:00Z, stop: 2017-12-31T12:00:00Z)' \
                             f'|> filter(fn: (r) => r._field == "leak_ground_truth")' \
                             f'|> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")'
        result_ground_truth = query_api.query_data_frame(org='primary', query=ground_truth_query)

        ground_truth = result_ground_truth['measurement']
        ground_truth.index = pd.to_datetime(result_ground_truth['_time'])

        pipelines = ['pipeline0', 'pipeline1', 'pipeline2', 'pipeline3']
        metrics = [f1_score, recall_score, precision_score]

        predictions_ix = pd.to_datetime(result['_time'])
        ground_truth_prediction_range = ground_truth[predictions_ix]

        for pipeline in pipelines:
            if pipeline in result.columns:
                pipeline_predictions = result[pipeline]
                pipeline_predictions.index = predictions_ix
                # Filtering for None values still needs finetuning
                pipeline_predictions = pipeline_predictions[pipeline_predictions != 'None']
                for metric in metrics:
                    performance = compute_performance_metric(
                        ground_truth_prediction_range, pipeline_predictions, metric)
                    print(f'{pipeline}: {metric}: {performance}')


if __name__ == '__main__':
    run()
