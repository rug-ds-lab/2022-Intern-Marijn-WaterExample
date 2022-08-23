import time

import os
import pandas as pd
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from configparser import ConfigParser
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, accuracy_score

KAFKA_SERVER = 'kafka:9092'


def compute_performance_metric(y_true, y_pred, metric):
    """
    Computes the performance based on a given metric

    :param y_true:
    :param y_pred:
    :param metric: metric function
    :return:
    """
    if metric == accuracy_score:
        performance = metric(y_true, y_pred)
    else:
        performance = metric(y_true, y_pred, zero_division=0)
    return performance


def run():
    config = ConfigParser()
    config.read('config.ini')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    while True:
        # Compute performance every two seconds
        time.sleep(2)

        # Query all generated predictions from database
        prediction_query = f' from(bucket:"primary") ' \
                           f'|> range(start: 2017-01-01T00:00:00Z, stop: 2017-12-31T12:00:00Z)' \
                           f'|> filter(fn: (r) => r._field == "binary_leak_prediction")' \
                           f'|> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")'
        result = query_api.query_data_frame(org='primary', query=prediction_query)

        # Query saved ground truths from database
        ground_truth_query = f' from(bucket:"primary") ' \
                             f'|> range(start: 2017-01-01T00:00:00Z, stop: 2017-12-31T12:00:00Z)' \
                             f'|> filter(fn: (r) => r._field == "leak_ground_truth")' \
                             f'|> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")'
        result_ground_truth = query_api.query_data_frame(org='primary', query=ground_truth_query)

        # Define constants
        pipelines = ['pipeline0', 'pipeline1', 'pipeline2', 'pipeline3']
        metrics = [f1_score, recall_score, precision_score, accuracy_score]
        metric_names = ['f1-score', 'recall-score', 'precision-score', 'accuracy_score']

        if len(result) > 0 and len(result_ground_truth) > 0:
            ground_truth = result_ground_truth['measurement']
            ground_truth.index = pd.to_datetime(result_ground_truth['_time'])

            predictions_ix = pd.to_datetime(result['_time'])

            # Compute all metrics for all pipelines
            for pipeline in pipelines:
                if pipeline in result.columns:
                    pipeline_predictions = result[pipeline]
                    pipeline_predictions.index = predictions_ix
                    pipeline_predictions = pipeline_predictions[pipeline_predictions.notna()].astype(bool)
                    ground_truth_prediction_range = ground_truth[pipeline_predictions.index]
                    for metric, metric_name in zip(metrics, metric_names):
                        performance = compute_performance_metric(
                            ground_truth_prediction_range, pipeline_predictions, metric)

                        # Save value to database
                        p = Point(pipeline) \
                            .tag('metric_name', metric_name) \
                            .field('metric_value', performance)
                        write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()
