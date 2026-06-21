# ایمیج سبک فقط برای فاز تست — فقط Endpoint /ask
FROM python:3.11-slim

WORKDIR /app

# نصب فقط وابستگی‌های سبک (بدون torch / transformers / faiss / ...)
COPY requirements-ask.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements-ask.txt

COPY app_ask.py .
COPY admin_panel.html .

EXPOSE 8000

CMD ["uvicorn", "app_ask:app", "--host", "0.0.0.0", "--port", "8000"]
