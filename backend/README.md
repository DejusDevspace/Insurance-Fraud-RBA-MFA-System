# Backend Service - Insurance Fraud Detection

This directory contains the core backend API for the Insurance Fraud Detection system. Built with Python and FastAPI, it handles user authentication, multi-factor authentication (MFA) flows, claims management, and interfaces with the pre-trained machine learning models for real-time risk assessment and fraud detection.

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.13)
- **Database**: PostgreSQL with [SQLAlchemy](https://www.sqlalchemy.org/) ORM (asyncpg & psycopg2)
- **Migrations**: [Alembic](https://alembic.sqlalchemy.org/)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Machine Learning Inference**: scikit-learn, XGBoost, SHAP, joblib

## Prerequisites

To run this project locally without Docker, you will need:

- Python >= 3.13
- `uv` (Fast Python package and project manager)
- A running PostgreSQL database instance

## Local Development Setup

1. **Install Dependencies**
   We use `uv` for lightning-fast dependency management, ensuring exact versions from the `uv.lock` file are used.

   ```bash
   # Navigate to the backend directory
   cd backend

   # Create a virtual environment and sync dependencies
   uv venv
   uv sync

   # Activate the virtual environment
   source .venv/bin/activate
   ```

2. **Environment Configuration**
   Copy the example environment variables and adjust the database credentials and secrets as necessary.

   ```bash
   cp .env.example .env
   ```

   _Note: Ensure your `DATABASE_URL` points to a valid local or remote PostgreSQL instance._

3. **Database Migrations**
   Initialize the database schema using Alembic.

   ```bash
   alembic upgrade head
   ```

4. **Run the Development Server**
   Start the FastAPI server with hot-reloading enabled.
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Documentation

Once the server is running, FastAPI automatically generates interactive documentation interfaces:

- **Swagger UI**: `http://localhost:8000/docs` (Great for testing endpoints directly)
- **ReDoc**: `http://localhost:8000/redoc` (Alternative, clean documentation format)

## Standalone Docker Deployment

The backend contains a production-ready `Dockerfile` optimized for performance. It acts as a standalone entry point for deploying the backend service.

To build and run the image independently:

```bash
# Make sure you are in the backend directory
cd backend

# Build the image
docker build -t insurance-fraud-backend .

# Run the container (ensure you pass necessary environment variables)
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e SECRET_KEY="your-secret-key" \
  insurance-fraud-backend
```

_Tip: The Dockerfile utilizes `uv` in a multi-stage build to ensure the smallest possible runtime footprint while adhering strictly to the lockfile._
