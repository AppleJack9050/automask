import os, json, sqlite3
import pika, requests
import uuid
import pika
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import reduce
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

        connection = pika.BlockingConnection(connection_params)
        self.channel = connection.channel()
        self.channel.queue_declare(queue=QUEUE, durable=True)

    def create_file_queue(self, user: str, files: list, prompt: str, positive: bool, highlight: bool, saved: bool):
        message = {
            "user_id": user,
            "files": files,
            "prompt": prompt,
            "positive": positive,
            "highlight": highlight,
            "saved": saved
        }

        message = json.dumps(message)
        self.channel.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2
        ))
