# Small official Python image
FROM python:3.12-slim

WORKDIR /code

# Install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and the model
COPY app ./app

EXPOSE 8000

# Listen on 0.0.0.0 so the container is reachable from outside
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]