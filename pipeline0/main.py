from kafka import KafkaConsumer


def run():
    kafka_servers = ['kafka:9093']
    consumer = KafkaConsumer(
        'dma_data',
        auto_offset_reset='earliest',
        group_id=None,
        enable_auto_commit=True,
        bootstrap_servers=kafka_servers
    )
    print('Successfully connected')
    for message in consumer:
        print(message.value)


if __name__ == '__main__':
    run()
