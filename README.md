# Foreign Language Tutor

A single web app for practicing spoken Spanish, with **two independent modes** you switch
between at the top of the page:

- **Raw Nova Sonic** — live speech-to-speech. You talk, the tutor talks back, streamed,
  with natural turn-taking and barge-in. Pure conversation: no pronunciation or grammar
  feedback, no tools. One model does STT + reasoning + TTS.
- **Custom bot** — record one utterance at a time and get the full treatment:
  transcription, phoneme-level **pronunciation** scoring, **grammar** feedback, a spoken
  reply, and course-material lookup (RAG). This is the project's differentiator.

There is no "hybrid" mode — the two are deliberately separate.

## Architecture

```
                         ┌──────────────  CloudFront  ──────────────┐
Browser (web app) ──────▶│  /            → S3 static site           │
                         │  /ws          → Fargate Nova Sonic bridge │  (Mode A)
                         │  /converse    → HTTP API → Lambda         │  (Mode B)
                         │  /upload-url  → HTTP API → Lambda         │
                         └───────────────────────────────────────────┘

Mode A — Raw Nova Sonic (live):
  Browser mic (PCM16 16k) ⇄ WebSocket ⇄ Fargate bridge ⇄ Bedrock Nova Sonic
  bidirectional stream  →  streamed speech back (PCM16 24k). Token-gated.

Mode B — Custom bot (per utterance):
  Browser records WAV → S3 (presigned PUT)
    → POST /converse → Lambda:
        → Amazon Transcribe (server-side STT)
        → SageMaker Wav2Vec2 (phoneme extraction)      [optional, -c phonemes=false]
        → Bedrock Claude Haiku (conversation + evaluation)
            → tool_use → Bedrock KB retrieve (course materials)
        → Polly Neural TTS (spoken reply → S3)
    → reply text + transcript + pronunciation/grammar notes + audio URL
```

## Deploy

Requires a container runtime for the image builds and the Lambda bundling step. `deploy.sh`
auto-selects Finch (or Docker) — use it instead of `npx cdk`.

```bash
pip install -r requirements.txt
./deploy.sh                      # full stack (includes the GPU phoneme endpoint)
./deploy.sh -c phonemes=false    # skip the 24/7 GPU endpoint; grammar feedback only
./deploy.sh synth                # any cdk subcommand passes through
```

Key stack outputs:

| Output | Use |
|--------|-----|
| `AppUrl` | Open this in a browser. |
| `WsUrl` | Raw-Sonic WebSocket (the web client fills this in from `config.json`). |
| `AccessTokenSecretName` | The shared secret gating Raw-Sonic sessions (see below). |
| `MaterialsBucketName` | S3 data source for the Knowledge Base. |
| `SyncFunctionName` | Force a Canvas sync via `aws lambda invoke`. |

### Get the Raw-Sonic access token

The Sonic WebSocket is gated by a generated shared secret. Paste it into the **Access
token** field in the web app (Raw Sonic mode):

```bash
aws secretsmanager get-secret-value \
  --secret-id language-tutor/bridge-access-token \
  --query SecretString --output text
```

### Post-deploy: Create the Knowledge Base (course materials)

1. Bedrock console → Knowledge Bases → Create.
2. Name `tutor-materials`, Quick Create, Titan Embeddings V2.
3. S3 data source → use `MaterialsBucketName` from the stack outputs.
4. Copy the KB ID and wire it in:

```bash
python3 setup_kb.py <KB_ID>
```

### Canvas LMS setup (optional)

```bash
aws secretsmanager put-secret-value --secret-id language-tutor/canvas-api-token \
  --secret-string '{"base_url":"https://your-school.instructure.com","token":"...","course_id":"12345"}'
```

Materials sync runs every 6 hours automatically. Force a sync:

```bash
aws lambda invoke --function-name <SyncFunctionName> /dev/null
```

## Security notes

- **Raw-Sonic WebSocket** (`/ws`) requires the shared access token on the upgrade
  (`?token=` or `Authorization: Bearer`).
- **Custom-bot API** (`/converse`, `/upload-url`) is currently **unauthenticated** —
  anyone with the URL can spend Bedrock/Transcribe/Polly on your bill. Add a Cognito/JWT
  or Lambda authorizer before exposing it publicly.

## CLI client (custom-bot mode)

The browser is the primary client. A CLI client for the custom-bot pipeline is also kept:

```bash
pip install amazon-transcribe pyaudio boto3 requests
python3 converse_client.py
```

## System prompts

- `prompts/system_sonic.txt` — Raw Nova Sonic (conversation only).
- `prompts/system.txt` — Custom bot (conversation + pronunciation/grammar evaluation +
  course-material tool).

## Project structure

```
app.py                     # CDK entry point
cdk/stack.py               # Unified stack: both backends + CloudFront hosting
frontend/                  # Web app (mode toggle, both flows)
  index.html  app.js  capture-worklet.js  styles.css
sonic_container/           # Mode A: raw Nova Sonic Fargate bridge
  server.py  Dockerfile  requirements.txt
lambda/
  handler.py               # Mode B: Transcribe + SageMaker + Bedrock + KB + Polly
  requirements.txt         # amazon-transcribe (bundled into the converse Lambda)
  sync.py                  # Canvas sync handler
container/                 # Mode B: Wav2Vec2 phoneme SageMaker container
  serve.py  Dockerfile
prompts/
  system.txt  system_sonic.txt
deploy.sh  bin/docker      # Finch/Docker deploy wrapper
converse_client.py         # CLI client (custom-bot mode)
setup_kb.py                # Post-deploy KB wiring
requirements.txt
```
