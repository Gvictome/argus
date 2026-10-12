FROM python:3.11-slim

WORKDIR /app

# Built from the repo root (see docker-compose.yml) so the robust
# aggregation code can come from src/federated rather than a copy.
COPY central_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY central_server/fl_aggregator.py .
COPY central_server/central_config.json .
COPY src/federated/robust.py src/federated/robust_strategy.py ./

RUN mkdir -p /data/checkpoints

ENV ARGUS_CONFIG=/app/central_config.json
ENV PYTHONUNBUFFERED=1

# FL gRPC server + REST API
EXPOSE 8080 8090

CMD ["python", "fl_aggregator.py"]
