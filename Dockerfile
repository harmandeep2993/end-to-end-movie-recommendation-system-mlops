# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# copy environment spec first for layer caching
COPY environment.yml /app/
RUN pip install --upgrade pip \
 && pip install numpy pandas scipy scikit-learn fastapi uvicorn[standard] joblib

# copy project
COPY src /app/src
COPY models /app/models
COPY data/processed /app/data/processed

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]