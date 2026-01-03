#!/bin/bash
# Initialize database with sample data (optional)

set -e

echo "Initializing database..."

# Wait for database to be ready
until PGPASSWORD=$DB_PASSWORD psql -h "postgres" -U "$DB_USER" -d "$DB_NAME" -c '\q'; do
  echo "Waiting for database to be ready..."
  sleep 1
done

echo "Database is ready!"

# Run migrations (if using Alembic)
echo "Running migrations..."
alembic upgrade head

echo "Database initialization complete!"
