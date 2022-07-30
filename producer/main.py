from kafka import KafkaProducer
import pandas as pd
import glob
import re
import os
from time import sleep
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
from time import mktime
from configparser import ConfigParser

KAFKA_SERVER = 'kafka:9092'


def read_data_as_matrix(folder_path, network_property):
    network_data = pd.DataFrame()

    if network_property == 'Pressures':
        file_name = 'Node_*.csv'
    elif network_property == 'Flows':
        file_name = 'Link_*.csv'
    else:
        raise Exception('Incorrect property')

    data_path = os.path.join(folder_path, network_property, file_name)

    print(data_path)

    for d in glob.glob(data_path):
        node_pressure = pd.read_csv(d)['Value']
        node_n = int(re.sub('\D', '', os.path.basename(d)))
        network_data[node_n] = node_pressure
    return network_data.reindex(sorted(network_data.columns), axis=1)


def run():
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER)
    print('Producer started')

    config = ConfigParser()
    config.read('config.ini')
    scenario_path = config['producer']['scenario_path']
    start_time = int(config['producer']['start_time'])

    pressures = read_data_as_matrix(folder_path=scenario_path, network_property='Pressures')
    flows = read_data_as_matrix(folder_path=scenario_path, network_property='Flows')
    print(pressures)

    timestamp_path = os.path.join(scenario_path, 'Timestamps.csv')

    timestamps = pd.read_csv(timestamp_path)

    # Scenario 13: Leak Start is at 6587
    pressures = pressures.iloc[start_time:, :].reset_index(drop=True)
    flows = flows.iloc[start_time:, :].reset_index(drop=True)
    timestamps = timestamps.iloc[start_time:, :].reset_index(drop=True)

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for p, f in zip(pressures.index, flows.index):
        current_pressure = pressures.iloc[p]
        print(timestamps.iloc[p])

        current_time = datetime\
            .fromisoformat(timestamps.iloc[p]['Timestamp'])\
            .timetuple()\

        unix_time = int(mktime(current_time))

        for node in current_pressure.index.astype('int64').sort_values():
            p = Point('measurement')\
                .tag('node', node)\
                .field('pressure', current_pressure[node])\
                .time(unix_time, write_precision='s')
            write_api.write(bucket='primary', record=p)

        current_flow = flows.iloc[f]

        print(current_flow.index.astype('int64').sort_values())

        for link in current_flow.index.astype('int64').sort_values():
            p = Point('measurement') \
                .tag('link', link) \
                .field('flow', current_flow[link]) \
                .time(unix_time, write_precision='s')
            write_api.write(bucket='primary', record=p)

        pressure_data = current_pressure
        pressure_data['time'] = unix_time
        pressure_data_json = pressure_data.to_json()

        producer.send('pressure-data', pressure_data_json.encode('utf-8'))

        flow_data = current_flow
        flow_data['time'] = unix_time
        flow_data_json = flow_data.to_json()

        producer.send('flow-data', flow_data_json.encode('utf-8'))

        producer.flush()
        sleep(0.5)


if __name__ == '__main__':
    run()
