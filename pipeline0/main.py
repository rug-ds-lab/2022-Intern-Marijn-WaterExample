from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

KAFKA_SERVER = 'kafka:9092'


def run():
    consumer = KafkaConsumer('dma-epynet_data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline0 started')

    client = InfluxDBClient(url='http://influxdb:8086', username='admin', password='bitnami123', org='primary')
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for msg in consumer:
        measurement = json.loads(msg.value.decode('utf-8'))
        print(measurement)

        p = Point('measurement').tag('node', '2').field('pressure', measurement['2'])
        # write_api.write(bucket='primary', record=p)


if __name__ == '__main__':
    run()

