# syntax=docker/dockerfile:1
# Use a lightweight Python 3.11 image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy environment file for dependency management (optional but useful for tracking)
COPY environment.yml /app/

# Upgrade pip and install required Python packages
# Using direct pip install here for speed instead of creating a Conda environment
RUN pip install --upgrade pip \
 && pip install numpy pandas scipy scikit-learn fastapi uvicorn[standard] joblib

# Copy the project source code, trained models, and processed data into the container
COPY src /app/src
COPY models /app/models
COPY data/processed /app/data/processed

# Expose port 8000 to allow access to the FastAPI application
EXPOSE 8000

# Command to start the FastAPI server using Uvicorn
# --host 0.0.0.0 makes it accessible outside the container
# --port 8000 defines the server port
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------
# HOW THE API CONNECTS:
# When this container runs, Uvicorn launches the FastAPI app defined in src/serving/app.py.
# The app listens on http://0.0.0.0:8000 inside the container.
# If you map the container port to your local machine (e.g., `docker run -p 8000:8000 <image>`),
# you can access the API at http://localhost:8000.
# FastAPI automatically exposes endpoints (like /predict) defined in your app.py or predict.py files.