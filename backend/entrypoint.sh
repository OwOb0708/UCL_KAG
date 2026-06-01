#!/bin/bash
set -e

# Substitute environment variables into kag_config.yaml template
envsubst < /app/kag_config.yaml.template > /app/kag_config.yaml
echo "[entrypoint] kag_config.yaml generated at /app/kag_config.yaml"

export KAG_CONFIG_PATH=/app/kag_config.yaml

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
