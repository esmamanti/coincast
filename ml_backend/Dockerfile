FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./requirements.txt
COPY ml_backend/requirements.txt ./ml_backend_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r ml_backend_requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "ml_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
