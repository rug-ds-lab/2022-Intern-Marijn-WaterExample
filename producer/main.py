from kafka import KafkaProducer

KAFKA_SERVER = 'kafka:9092'


def run():
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER)
    print('Producer started')

    for _ in range(0, 20):
        producer.send('dma-data', b'message')
        producer.flush()


if __name__ == '__main__':
    run()
