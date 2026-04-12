FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fl_aggregator.py .
COPY central_config.json .

RUN mkdir -p /data/checkpoints

ENV ARGUS_CONFIG=/app/central_config.json
ENV PYTHONUNBUFFERED=1

# FL gRPC server + REST API
EXPOSE 8080 8090

CMD ["python", "fl_aggregator.py"]
