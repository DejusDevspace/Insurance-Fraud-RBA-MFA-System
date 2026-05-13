# Insurance Fraud Detection System with Risk-Based Authentication (RBA) & MFA

An intelligent, end-to-end insurance system designed to detect fraudulent claims and assess user risk in real-time, enforcing Multi-Factor Authentication (MFA) based on contextual risk scores.

## Architecture Overview

This project is divided into three main components, each housed in its respective directory:
- **`backend/`**: A high-performance Python backend built with FastAPI. It handles core business logic, user authentication, claim processing, and serves the machine learning models.
- **`frontend/`**: A modern web interface built with React 19, TypeScript, and Tailwind CSS v4, compiled using Vite. It provides dashboards for administrators and portals for users.
- **`ml/`**: The data science pipeline. It contains Python scripts for generating synthetic insurance data, training Risk Assessment and Fraud Detection models using scikit-learn and XGBoost, and generating SHAP explainers.

## Getting Started (Quickstart)

The easiest way to spin up the entire application is using Docker and Docker Compose. We provide a `Makefile` to simplify common operations.

### Prerequisites
- Docker & Docker Compose
- `make` (optional, but recommended)

### Running the Stack
1. Clone the repository and navigate to the root directory.
2. Ensure you have an environment file by copying the example (if applicable):
   ```bash
   cp .env.example .env
   ```
3. Start all services using Make:
   ```bash
   make up
   # or run standard docker-compose if Make is unavailable:
   # docker-compose up -d
   ```

Once running, the services will be available at:
- **Frontend Dashboard**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`
- **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`

## Makefile Commands Reference

We use a Makefile to orchestrate common development tasks. Run `make help` to see all available commands. Some useful commands include:

- `make dev` - Start the development environment (docker-compose up -d).
- `make down` - Stop all running services.
- `make logs` - View logs from all services in real-time.
- `make shell-backend` - Open an interactive bash shell inside the backend container.
- `make shell-db` - Open a `psql` shell into the PostgreSQL database.
- `make migrations` - Run Alembic database migrations.
- `make backup` / `make restore` - Manage database snapshots.
- `make prod` - Start the production environment using `docker-compose.prod.yml`.

## Detailed Documentation

For an in-depth understanding of the system's design and codebase, please refer to the following documents:
- [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md): Overview of business requirements, system architecture, and core features.
- [CODEBASE_DOCUMENTATION.md](./CODEBASE_DOCUMENTATION.md): Deep dive into technical implementation details, API schemas, and ML model specs.

For component-specific setup and development guides, refer to the README files located inside each directory:
- [Backend Documentation](./backend/README.md)
- [Frontend Documentation](./frontend/README.md)
- [Machine Learning Documentation](./ml/README.md)
