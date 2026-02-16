FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p drops/processed drops/failed data

EXPOSE 8000

CMD ["uvicorn", "emslite.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
