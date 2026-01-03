#!/bin/bash
# Database backup script for production PostgreSQL

set -e

BACKUP_DIR="/backups"
DB_HOST="postgres"
DB_USER="fraud_user"
DB_NAME="insurance_fraud"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/insurance_fraud_${TIMESTAMP}.sql"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "Starting database backup at $(date)"

# Create backup
pg_dump -h "$DB_HOST" -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"
BACKUP_FILE="${BACKUP_FILE}.gz"

echo "Backup completed: $BACKUP_FILE"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "insurance_fraud_*.sql.gz" -mtime +7 -delete

echo "Backup script finished at $(date)"
