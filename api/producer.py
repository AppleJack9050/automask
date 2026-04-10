import os, json
import pika
import pika
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")
IMAGE_DB = os.getenv("IMAGE_DB")
QUEUE = "image_queue"

class Producer():
    def __init__(self):
        self.channel = None
        
        try:
            self.setup_rabbitmq_producer()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection failed: {e}, retrying in 5s...")
            time.sleep(5)
            self.setup_rabbitmq_producer()

    def setup_rabbitmq_producer(self):
        host = os.getenv("RABBITMQ_HOST", "localhost")
        connection_params = pika.ConnectionParameters(
            host=host,
            heartbeat=0
        )

        while True:
            try:
                connection = pika.BlockingConnection(connection_params)
                self.channel = connection.channel()
                self.channel.queue_declare(queue=QUEUE, durable=True)
                break
            except pika.exceptions.AMQPConnectionError:
                print("Retrying RabbitMQ...")
                time.sleep(5)


    def create_file_queue(self, user: str, files: list, prompt: str, positive: bool, highlight: bool, saved: bool):
        max_files_per_message = 50
        messages = []

        for i in range(0, len(files), max_files_per_message):
            chunk = files[i:i + max_files_per_message]
            message = {
                "user_id": user,
                "files": chunk,
                "prompt": prompt,
                "positive": positive,
                "highlight": highlight,
                "saved": saved
            }
            messages.append(message)

        for message in messages:
            message = json.dumps(message)
            self.channel.basic_publish(
                exchange="",
                routing_key=QUEUE,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2
            ))
