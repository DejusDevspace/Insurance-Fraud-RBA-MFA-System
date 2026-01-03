.PHONY: help build up down logs restart clean test dev prod ps shell-backend shell-db

help:
	@echo "Insurance Fraud Detection System - Makefile Commands"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev              - Start development environment"
	@echo "  make up               - Start all services"
	@echo "  make down             - Stop all services"
	@echo "  make logs             - View logs from all services"
	@echo "  make restart          - Restart all services"
	@echo "  make ps               - List running containers"
	@echo ""
	@echo "Database Commands:"
	@echo "  make shell-db         - Connect to PostgreSQL shell"
	@echo "  make backup           - Create database backup"
	@echo "  make restore          - Restore from backup (requires BACKUP_FILE)"
	@echo ""
	@echo "Backend Commands:"
	@echo "  make shell-backend    - Shell into backend container"
	@echo "  make backend-logs     - View backend logs"
	@echo "  make migrations       - Run database migrations"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build            - Build Docker images"
	@echo "  make build-no-cache   - Build without cache"
	@echo "  make clean            - Remove containers, networks, volumes"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test             - Run tests"
	@echo "  make test-backend     - Run backend tests only"
	@echo ""
	@echo "Production Commands:"
	@echo "  make prod             - Start production environment"
	@echo "  make prod-down        - Stop production environment"
	@echo ""
	@echo "Examples:"
	@echo "  make dev              # Start development"
	@echo "  make prod             # Start production"
	@echo "  make shell-db         # Enter database"
	@echo "  make backup           # Backup database"

# Development
dev:
	docker-compose up -d
	@echo "✓ Development environment started"
	@echo "  Frontend: http://localhost:5173"
	@echo "  Backend: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

up:
	docker-compose up -d
	@echo "✓ All services started"

down:
	docker-compose down
	@echo "✓ All services stopped"

logs:
	docker-compose logs -f

restart:
	docker-compose restart
	@echo "✓ All services restarted"

ps:
	docker-compose ps

clean:
	docker-compose down -v
	@echo "✓ Containers, networks, and volumes removed"

# Backend
shell-backend:
	docker-compose exec backend bash

backend-logs:
	docker-compose logs -f backend

migrations:
	docker-compose exec backend alembic upgrade head
	@echo "✓ Migrations completed"

# Database
shell-db:
	docker-compose exec postgres psql -U fraud_user -d insurance_fraud

backup:
	docker-compose exec postgres pg_dump -U fraud_user insurance_fraud > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✓ Database backed up"

restore:
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Usage: make restore BACKUP_FILE=backups/backup_XXXXX.sql"; exit 1; fi
	docker-compose exec -T postgres psql -U fraud_user insurance_fraud < $(BACKUP_FILE)
	@echo "✓ Database restored from $(BACKUP_FILE)"

# Docker
build:
	docker-compose build
	@echo "✓ Images built"

build-no-cache:
	docker-compose build --no-cache
	@echo "✓ Images built without cache"

# Testing
test:
	@echo "Note: Configure test database in pytest.ini"
	docker-compose exec backend pytest

test-backend:
	docker-compose exec backend pytest app/tests

# Production
prod:
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✓ Production environment started"

prod-down:
	docker-compose -f docker-compose.prod.yml down
	@echo "✓ Production environment stopped"

prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f
