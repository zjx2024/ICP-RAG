#!/bin/bash

wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_retries=30
    local retry_count=0
    
    echo "Waiting for $service_name at $host:$port..."
    
    while ! nc -z "$host" "$port"; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "ERROR: $service_name not available after $max_retries attempts"
            exit 1
        fi
        sleep 2
    done
    
    echo "$service_name is ready!"
}

wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT" "PostgreSQL"
wait_for_service "$ES_HOST" "$ES_PORT" "Elasticsearch"
wait_for_service "$HUGEGRAPH_HOST" "$HUGEGRAPH_PORT" "HugeGraph"

echo "All services are ready. Starting application..."

exec uvicorn backend.main:app --host 0.0.0.0 --port 9999 --log-level info