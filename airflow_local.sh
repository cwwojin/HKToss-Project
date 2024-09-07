#!/usr/bin/env bash

docker compose -f docker-compose.mlflow.yml up -d

PYTHON_ENV=development docker compose --profile dev -f docker-compose.airflow.yml up -d --force-recreate