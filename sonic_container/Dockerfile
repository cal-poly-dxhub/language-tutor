FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8080

# Nova Sonic bidirectional streaming needs the container's task-role credentials to be
# resolvable from the environment. The ECS credential provider populates AWS_* env vars,
# which the experimental SDK's EnvironmentCredentialsResolver reads.
ENTRYPOINT ["python", "server.py"]
