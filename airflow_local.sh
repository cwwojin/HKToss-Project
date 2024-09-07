#!/usr/bin/env bash

PYTHON_ENV=development docker compose --profile dev -f docker-compose.airflow.yml up -d --force-recreate