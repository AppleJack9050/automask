import os, json, sqlite3
import pika, requests
import uuid
import pika
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import reduce

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")
IMAGE_DB = os.getenv("IMAGE_DB")
QUEUE = "image_queue"

class Producer():
    def __init__(self):
        self.channel = None
        self.setup_rabbitmq_producer()
        
    def setup_rabbitmq_producer(self):
        host = os.getenv("RABBITMQ_HOST", "localhost")
        connection_params = pika.ConnectionParameters(
            host=host,
            heartbeat=0
        )

        connection = pika.BlockingConnection(connection_params)
        self.channel = connection.channel()
        self.channel.queue_declare(queue=QUEUE, durable=True)

    def check_queue(self) -> list:
        files = []
        while True:
            method, properties, body = self.channel.basic_get(queue=QUEUE, auto_ack=False)
            if method is None:
                break
            body = json.loads(body)
            files.append(body["files"])
            self.channel.basic_nack(method.delivery_tag, requeue=True)
        return reduce(lambda x, y: x + y, files, [])

    def create_file_queue(self, files, prompt, positive, highlight):
        message = {
#            "user_id": q.user_id,
            "files": files,
            "prompt": prompt,
            "positive": positive,
            "highlight": highlight
        }

        message = json.dumps(message)
        self.channel.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2
        ))
