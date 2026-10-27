FROM python:3.12-slim
RUN pip install --no-cache-dir confluent-kafka==2.6.1
COPY producer.py /app/producer.py
CMD ["python", "-u", "/app/producer.py"]
