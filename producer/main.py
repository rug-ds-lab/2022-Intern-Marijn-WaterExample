from kafka import KafkaProducer
import pandas as pd
import os
from time import sleep
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from time import mktime
from configparser import ConfigParser

KAFKA_SERVER = 'kafka:9092'


def run():
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, api_version=(0, 10, 1))

    config = ConfigParser()
    config.read('config.ini')
    scenario_path = config['global']['scenario_path']

    parameters = ConfigParser()
    parameters.read(os.path.join(scenario_path, 'parameters.ini'))

    start_time = config.get('global', 'experiment_start_time')
    message_frequency = config.getfloat('global', 'message_frequency')

    labels = pd.read_csv(
        os.path.join(scenario_path, 'labels.csv'),
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    pressures = pd.read_csv(
        os.path.join(scenario_path, 'pressure', 'test.csv'),
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )
    flows = pd.read_csv(
        os.path.join(scenario_path, 'flow', 'test.csv'),
        infer_datetime_format=True,
        parse_dates=True,
        index_col=0
    )

    pressures = pressures[start_time:]
    flows = flows[start_time:]

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for t in pressures.index:
        print(t)

        current_pressure = pressures.loc[t]
        current_flow = flows.loc[t]
        current_time = t.to_pydatetime()

        unix_time = int(mktime(current_time.timetuple()))

        p = Point('measurement') \
            .field('leak_ground_truth', bool(labels.loc[t][0])) \
            .time(unix_time, write_precision='s')
        write_api.write(bucket='primary', record=p)

        for node in current_pressure.index:
            p = Point('measurement')\
                .tag('node', node)\
                .field('pressure', current_pressure[node])\
                .time(unix_time, write_precision='s')
            write_api.write(bucket='primary', record=p)

        for link in current_flow.index:
            p = Point('measurement') \
                .tag('link', link) \
                .field('flow', current_flow[link]) \
                .time(unix_time, write_precision='s')
            write_api.write(bucket='primary', record=p)

        pressure_data = current_pressure
        pressure_data['time'] = unix_time
        pressure_data_json = pressure_data.to_json()

        flow_data = current_flow
        flow_data['time'] = unix_time
        flow_data_json = flow_data.to_json()

        producer.send('pressure-data', pressure_data_json.encode('utf-8'))
        producer.send('flow-data', flow_data_json.encode('utf-8'))

        producer.flush()
        sleep(message_frequency)


if __name__ == '__main__':
    run()
