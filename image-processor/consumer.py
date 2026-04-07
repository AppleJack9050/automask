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
        self.connection = None
        self.file_processor = FileProcessor(
            os.getenv("UPLOAD_DIR"),
            os.getenv("PROCESSED_DIR"),
            os.getenv("SAVED_DIR")   
        )
        self.setup_consumer()

    def setup_consumer(self):
        host = os.getenv("RABBITMQ_HOST", "localhost")
        connection_params = pika.ConnectionParameters(
            host=host
        )
        self.connection = pika.BlockingConnection(connection_params)
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=QUEUE, durable=True) 

        self.channel.basic_consume(
            queue=QUEUE,
            on_message_callback=self.process,
            auto_ack=False
        )
        print("Image Processor Ready")
        self.channel.start_consuming()

    def process(self, ch, method, _, body):
        try:
            data = json.loads(body.decode('utf-8'))
            self.file_processor.create_process_queue(data["user_id"], data["files"])
            self.file_processor.process_files_in_queue(
                data["user_id"],
                data["prompt"],
                data["positive"],
                data["highlight"],
                data["saved"]
            )

            self.channel.basic_ack(delivery_tag=method.delivery_tag)
            return
        except Exception as e:
            print(f"Error processing message: {e}")
            self.channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

if __name__ == "__main__":
    import time
    while True:
        try:
            print("Starting Image Processor...")
            consumer = Consumer()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection failed: {e}, retrying in 5s...")
            time.sleep(5)
