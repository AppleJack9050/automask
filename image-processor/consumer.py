import os, json, sqlite3
import pika, requests
from fileProcessor import FileProcessor

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")
IMAGE_DB = os.getenv("IMAGE_DB")
QUEUE = "image_queue"

class Consumer():
    def __init__(self):
        self.channel = None
        self.setup_consumer()

    def setup_consumer(self):
        host = os.getenv("RABBITMQ_HOST", "localhost")
        connection_params = pika.ConnectionParameters(
            host=host
        )
        connection = pika.BlockingConnection(connection_params)
        self.channel = connection.channel()
        self.channel.queue_declare(queue=QUEUE, durable=True) 

        self.channel.basic_consume(
            queue=QUEUE,
            on_message_callback=self.process,
            auto_ack=False
        )

        self.channel.start_consuming()

    def process(self, ch, method, _, body):
        data = json.loads(body.decode('utf-8'))
        file_processor = FileProcessor(os.getenv("UPLOAD_DIR"), os.getenv("PROCESSED_DIR"))
        file_processor.create_process_queue(data["files"])
        file_processor.process_files_in_queue(
            data["prompt"],
            data["positive"],
            data["highlight"],
        )

        self.channel.basic_ack(delivery_tag=method.delivery_tag)
        return

if __name__ == "__main__":
    print("Starting worker...")
    consumer = Consumer()
