FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  libgomp1\
  && rm -rf /var/lib/apt/lists/*

COPY credit-scoring/requirements.txt .
COPY src/app.py .
COPY model/ ./model/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1234

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1234"]


