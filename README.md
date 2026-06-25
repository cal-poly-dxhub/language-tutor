# Foreign Language Tutor

Conversational Spanish tutor with pronunciation/grammar feedback and RAG-powered course material access.

## Architecture

```
Client (mic) → Transcribe Streaming (real-time STT)
             → S3 (audio upload via presigned URL)
             → HTTP API POST /converse
               → Lambda:
                 → SageMaker Wav2Vec2 (phoneme extraction)
                 → Bedrock Claude Haiku 4.5 (conversation + evaluation)
                   → tool_use → Bedrock KB retrieve (course materials)
                 → Polly Neural TTS (spoken response → S3)
             → Client (downloads + plays response audio)
```

## Deploy

```bash
pip install -r requirements.txt
CDK_DOCKER=finch npx cdk deploy
```

### Post-deploy: Create Knowledge Base

1. Open Bedrock console → Knowledge Bases → Create
2. Name: `tutor-materials`, Quick Create, Titan Embeddings V2
3. S3 data source → use the `MaterialsBucketName` from stack outputs
4. Copy the KB ID, then run:

```bash
python3 setup_kb.py <KB_ID>
```

### Canvas LMS Setup (optional)

Store Canvas API credentials in Secrets Manager:
```bash
aws secretsmanager put-secret-value --secret-id tutor/canvas-api-token \
  --secret-string '{"base_url":"https://your-school.instructure.com","token":"...","course_id":"12345"}'
```

Materials sync runs every 6 hours automatically, or force sync:
```bash
curl -X POST <SyncUrl>
```

## Client

```bash
pip install amazon-transcribe pyaudio boto3 requests
python3 converse_client.py
```

Speak Spanish → bot responds in Spanish (audio + text). Ask about class materials in English → bot searches KB and answers in English.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /upload-url` | Presigned S3 URL for audio upload |
| `POST /converse` | `{"text": "...", "audio_key": "...", "history": [...]}` → response + audio URL |
| `POST /sync` | Force Canvas → S3 → KB sync |

## System Prompt

Edit `prompts/system.txt` to change bot personality, evaluation strictness, or dialect tolerance.

## Project Structure

```
app.py                  # CDK entry point
cdk/stack.py            # Infrastructure
lambda/
  handler.py            # Conversation handler (SageMaker + Bedrock + KB + Polly)
  sync.py               # Canvas sync handler
container/
  serve.py              # SageMaker Wav2Vec2 container
  Dockerfile
prompts/system.txt      # System prompt
converse_client.py      # CLI client
setup_kb.py             # Post-deploy KB wiring
requirements.txt
```
