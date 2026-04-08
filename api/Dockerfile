FROM python:3.10-slim

WORKDIR /api
COPY . /api

RUN apt-get update && apt-get install -y curl openssl

RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

RUN python3 -m pip install -r requirements.txt
RUN mkdir -p /database /uploads /processed /saved
RUN chown -R apiuser:apiuser  . /database /uploads /processed /saved
USER apiuser

EXPOSE 80/tcp
CMD ["uvicorn", "endpoints:app", "--app-dir", "/api", "--host", "0.0.0.0", "--port", "80"]
