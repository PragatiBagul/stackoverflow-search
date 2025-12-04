# Dockerfile.api
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# copy repo
COPY . /app

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "src.stacksearch.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
