FROM ubuntu:22.04

WORKDIR /api
COPY . /api

RUN apt-get update && apt-get install -y python3 python3-pip 

RUN groupadd -r apiuser && useradd -r -g apiuser apiuser
RUN pip install -r requirements.txt
RUN  chown -R apiuser:apiuser .
USER apiuser

EXPOSE 80/tcp
CMD ["uvicorn", "endpoints:app", "--host", "0.0.0.0", "--port", "80" ]