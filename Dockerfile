# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY data/processed/ ./data/processed/
COPY data/raw/ ./data/raw/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]