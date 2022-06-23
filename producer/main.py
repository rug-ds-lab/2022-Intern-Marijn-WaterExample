from kafka import KafkaProducer
from time import sleep


def run():
    kafka_servers = ['kafka:9093']
    print('Attempting to connect to: ', kafka_servers)
    producer = KafkaProducer(bootstrap_servers=kafka_servers)
    print('Successfully connected')
    for _ in range(100):
        producer.send('dma_data', b'test')


if __name__ == '__main__':
    sleep(2)
    run()
