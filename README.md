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

## Alternative Architecture: Nova Sonic (speech-to-speech)

`NovaSonicTutorStack` (in `cdk/nova_sonic_stack.py`) is a second, independent stack that
replaces the STT → LLM → TTS pipeline with a single Amazon Nova Sonic speech-to-speech
model. Deploy it to compare against the pipeline above.

```
Browser (mic) ──ws──► ALB ──► Fargate bridge (nova_sonic_container/server.py)
                                 │  InvokeModelWithBidirectionalStream (HTTP/2)
                                 ▼
                          Amazon Nova Sonic  ── STT + reasoning + TTS in ONE model
                                 │             native turn-taking + barge-in
                                 ├─ tool: search_materials → Bedrock KB retrieve
                                 └─ tool: log_feedback → on-screen note (NOT spoken)
                                 ▼
Browser (speaker) ◄──ws── audio + transcript + feedback notes
```

Key differences vs. the pipeline:

| | Pipeline (`PronunciationCheckerStack`) | Nova Sonic (`NovaSonicTutorStack`) |
|---|---|---|
| Speech | Transcribe + Polly (2 services) | one model does STT + TTS |
| Turn-taking / barge-in | build it yourself | native |
| Pronunciation feedback | phoneme-precise (Wav2Vec2) | qualitative only (no phoneme output) |
| Compute | serverless Lambda | always-on Fargate (bidirectional stream can't run on Lambda) |
| GPU endpoint | SageMaker `ml.g4dn.xlarge` 24/7 | none |

The **"feedback without interrupting"** requirement is delivered by the `log_feedback`
tool: instead of speaking corrections, the model calls the tool and the bridge forwards
them to the client as text notes while the spoken conversation keeps flowing.

Deploy just this stack:

```bash
pip install -r requirements.txt
# Nova Sonic is region-limited; default region is us-east-1 (override with NOVA_SONIC_CDK_REGION)
CDK_DOCKER=finch npx cdk deploy NovaSonicTutorStack
python3 setup_kb.py <KB_ID>   # then set KNOWLEDGE_BASE_ID on the Fargate task + sync Lambda
```

> Note: the `WsUrl` output is `ws://` (no TLS). Browsers block mic access and `ws://`
> from `https://` pages — put an ACM cert + HTTPS listener on the ALB for `wss://` in
> production. The bridge uses the experimental `aws_sdk_bedrock_runtime` Python SDK
> (see `nova_sonic_container/requirements.txt`); Nova Sonic also has an ~8 min
> connection cap (renew/continue for longer sessions).

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
