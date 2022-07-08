from kafka import KafkaProducer
import pandas as pd
import glob
import re
import os
from time import sleep

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

    for i in pressures.index:
        current_pressure = pressures.iloc[i].to_json()
        producer.send('dma-data', current_pressure.encode('utf-8'))
        producer.flush()
        sleep(3)


if __name__ == '__main__':
    run()
