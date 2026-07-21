FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8080

# Nova Sonic bidirectional streaming needs the container's task role credentials to be
# resolvable from the environment. The ECS credential provider populates these via the
# container credentials endpoint; the experimental SDK's EnvironmentCredentialsResolver
# reads standard AWS_* env vars set by the ECS agent.
ENTRYPOINT ["python", "server.py"]
