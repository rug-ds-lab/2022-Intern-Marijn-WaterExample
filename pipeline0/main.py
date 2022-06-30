from kafka import KafkaConsumer

KAFKA_SERVER = 'kafka:9092'


def run():
    consumer = KafkaConsumer('dma-data', bootstrap_servers=KAFKA_SERVER)
    print('pipeline0 started')

    for msg in consumer:
        print(msg)


if __name__ == '__main__':
    run()

