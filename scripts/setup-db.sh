#!/usr/bin/env bash
# Start the local AIML PostgreSQL instance with pgvector.
set -euo pipefail

cd "$(dirname "$0")/.."
name=${AIML_POSTGRES_CONTAINER:-aiml-postgres}
port=${AIML_POSTGRES_PORT:-5433}
volume=${AIML_POSTGRES_VOLUME:-aiml_postgres_data}
password=${AIML_POSTGRES_PASSWORD:-aiml-local-dev}

if docker container inspect "$name" >/dev/null 2>&1; then
  docker start "$name" >/dev/null
else
  docker run --name "$name" \
    -e POSTGRES_USER=aiml -e POSTGRES_PASSWORD="$password" -e POSTGRES_DB=aiml_el \
    -p "127.0.0.1:$port:5432" -v "$volume:/var/lib/postgresql/data" \
    -d pgvector/pgvector:pg17
fi

for ((attempt=1; attempt<=30; attempt++)); do
  if docker exec "$name" pg_isready -h 127.0.0.1 -U aiml -d aiml_el >/dev/null 2>&1; then
    docker port "$name" 5432/tcp
    echo "PostgreSQL is ready. Backend startup applies the schema."
    exit 0
  fi
  sleep 1
done
echo "PostgreSQL did not become ready. Inspect the container logs." >&2
exit 1
