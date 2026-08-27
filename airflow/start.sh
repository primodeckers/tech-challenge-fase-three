#!/bin/bash
set -e
airflow db migrate
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@local \
  --password admin \
  2>/dev/null || airflow users reset-password --username admin --password admin
airflow scheduler &
exec airflow webserver --port 8080
