FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Listens on localhost only: this runs beside the bridge in the same task, and nothing
# outside that task has any business reaching it.
EXPOSE 8081

RUN useradd --create-home --uid 10002 asr
USER asr

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/health').status==200 else 1)"

ENTRYPOINT ["python", "-u", "app.py"]
