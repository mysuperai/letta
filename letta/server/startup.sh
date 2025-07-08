#!/bin/sh
set -e  # Exit on any error

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8283}"
LETTA_CONFIG_DIR="${LETTA_CONFIG_DIR:-}"

# Function to load configuration from mounted volume
load_config() {
    if [ -n "$LETTA_CONFIG_DIR" ] && [ -d "$LETTA_CONFIG_DIR" ]; then
        echo "Loading configuration from $LETTA_CONFIG_DIR..."
        
        # Load environment variables from .env file if it exists
        if [ -f "$LETTA_CONFIG_DIR/.env" ]; then
            echo "Loading environment variables from $LETTA_CONFIG_DIR/.env"
            set -a  # Automatically export variables
            . "$LETTA_CONFIG_DIR/.env"
            set +a
        fi
        
        # Copy custom alembic.ini if it exists
        if [ -f "$LETTA_CONFIG_DIR/alembic.ini" ]; then
            echo "Using custom alembic configuration from $LETTA_CONFIG_DIR/alembic.ini"
            cp "$LETTA_CONFIG_DIR/alembic.ini" /app/alembic.ini
        fi
        
        # Set custom OpenTelemetry config path if it exists
        if [ -f "$LETTA_CONFIG_DIR/otel-config.yaml" ]; then
            echo "Using custom OpenTelemetry configuration from $LETTA_CONFIG_DIR/otel-config.yaml"
            export CUSTOM_OTEL_CONFIG="$LETTA_CONFIG_DIR/otel-config.yaml"
        fi
        
        echo "Configuration loading completed."
    else
        echo "No custom configuration directory found or LETTA_CONFIG_DIR not set."
    fi
}

# Function to wait for PostgreSQL to be ready
wait_for_postgres() {
    until pg_isready -U "${POSTGRES_USER:-letta}" -h localhost; do
        echo "Waiting for PostgreSQL to be ready..."
        sleep 2
    done
}

# Load configuration from mounted volume first
load_config

# Check if we're configured for external Postgres
if [ -n "$LETTA_PG_URI" ]; then
    echo "External Postgres configuration detected, using env var LETTA_PG_URI"
else
    echo "No external Postgres configuration detected, starting internal PostgreSQL..."
    # Start PostgreSQL using the base image's entrypoint script
    /usr/local/bin/docker-entrypoint.sh postgres &

    # Wait for PostgreSQL to be ready
    wait_for_postgres

    # Set default connection URI for internal postgres
    export LETTA_PG_URI="postgresql://${POSTGRES_USER:-letta}:${POSTGRES_PASSWORD:-letta}@localhost:5432/${POSTGRES_DB:-letta}"
    echo "Using internal PostgreSQL at: $LETTA_PG_URI"
fi

# Attempt database migration
echo "Attempting to migrate database..."
if ! alembic upgrade head; then
    echo "ERROR: Database migration failed!"
    echo "Please check your database connection and try again."
    echo "If the problem persists, check the logs for more details."
    exit 1
fi
echo "Database migration completed successfully."

# Set permissions for tool execution directory if configured
if [ -n "$LETTA_SANDBOX_MOUNT_PATH" ]; then
    if ! chmod 777 "$LETTA_SANDBOX_MOUNT_PATH"; then
        echo "ERROR: Failed to set permissions for tool execution directory at: $LETTA_SANDBOX_MOUNT_PATH"
        echo "Please check that the directory exists and is accessible"
        exit 1
    fi
fi

# If ADE is enabled, add the --ade flag to the command
CMD="letta server --host $HOST --port $PORT"
if [ "${SECURE:-false}" = "true" ]; then
    CMD="$CMD --secure"
fi

# Start OpenTelemetry Collector in the background
if [ -n "$CUSTOM_OTEL_CONFIG" ]; then
    echo "Starting OpenTelemetry Collector with custom configuration..."
    CONFIG_FILE="$CUSTOM_OTEL_CONFIG"
elif [ -n "$CLICKHOUSE_ENDPOINT" ] && [ -n "$CLICKHOUSE_PASSWORD" ]; then
    echo "Starting OpenTelemetry Collector with Clickhouse export..."
    CONFIG_FILE="/etc/otel/config-clickhouse.yaml"
else
    echo "Starting OpenTelemetry Collector with file export only..."
    CONFIG_FILE="/etc/otel/config-file.yaml"
fi

/usr/local/bin/otelcol-contrib --config "$CONFIG_FILE" &
OTEL_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "Shutting down..."
    kill $OTEL_PID
    wait $OTEL_PID
}
trap cleanup EXIT

echo "Starting Letta Server at http://$HOST:$PORT..."
echo "Executing: $CMD"
exec $CMD
