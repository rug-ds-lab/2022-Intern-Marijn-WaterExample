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

KAFKA_SERVER = 'kafka:9092'


# noinspection DuplicatedCode
def read_pressure_as_matrix(folder_path):
    pressures = pd.DataFrame()
    for link_path in glob.glob(folder_path + '/Node_*.csv'):
        node_pressure = pd.read_csv(link_path)['Value']
        node_n = int(re.sub('\D', '', os.path.basename(link_path)))
        pressures[node_n] = node_pressure
    return pressures.reindex(sorted(pressures.columns), axis=1)


def run():
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER)
    print('Producer started')
    data_path = os.environ.get('DATA_PATH')
    scenario = 'Scenario-13/'
    values = 'Pressures'
    print(data_path + scenario)
    pressures = read_pressure_as_matrix(folder_path=data_path + scenario + values)
    print(pressures)
    timestamps = pd.read_csv(data_path + scenario + 'Timestamps.csv')

    # Scenario 13: Leak Start is at 6587
    pressures = pressures.iloc[6500:, :].reset_index(drop=True)
    timestamps = timestamps.iloc[6500:, :].reset_index(drop=True)

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for i in pressures.index:
        current_pressure = pressures.iloc[i]
        print(timestamps.iloc[i])

        current_time = datetime\
            .fromisoformat(timestamps.iloc[i]['Timestamp'])\
            .timetuple()\

        unix_time = int(mktime(current_time))

        for node in current_pressure.index:
            p = Point('measurement')\
                .tag('node', node)\
                .field('pressure', current_pressure[node])\
                .time(unix_time, write_precision='s')
            write_api.write(bucket='primary', record=p)

        current_pressure['time'] = unix_time
        current_pressure_json = current_pressure.to_json()
        producer.send('dma-epynet_data', current_pressure_json.encode('utf-8'))
        producer.flush()
        sleep(0.5)


if __name__ == '__main__':
    run()
