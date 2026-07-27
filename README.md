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

## Streaming Client (no press-Enter)

```bash
pip install amazon-transcribe pyaudio boto3 requests websockets webrtcvad
python converse_stream_client.py
```

Same conversation, but turn-taking is automatic and the reply is streamed:

- **No Enter key.** Voice-activity detection (VAD) ends your turn after a short
  pause (`SILENCE_MS` in the client, default 900ms).
- **Streamed reply.** Text appears token-by-token and audio plays
  sentence-by-sentence as it's synthesized — playback starts before the full
  reply is generated.

This runs over a WebSocket API and mirrors the protocol a browser client will use
(see *Scalability & web path* below). The original HTTP client above is unchanged.

### How the streaming path works

```
mic → VAD auto-endpoint (no Enter) + Transcribe streaming (live text)
    → full utterance uploaded to S3 (GET /upload-url)
    → WebSocket "converse" ─► StreamHandler Lambda:
         SageMaker phonemes → Bedrock token stream
           ├─ <reply> text → sentence chunks → Polly → audio pushed as it's ready
           └─ pronunciation/grammar notes pushed after the stream
    ← reply_delta / audio / *_note / done  (played + printed incrementally)
```

WebSocket message protocol (`lambda/stream_handler.py`):

| Direction | Message |
|---|---|
| client → server | `{"action":"converse","text","audio_key","history"}` |
| server → client | `{"type":"status","stage":"phonemes"\|"thinking"}` |
| server → client | `{"type":"reply_delta","text":"…"}` |
| server → client | `{"type":"audio","seq":n,"format":"mp3","data":"<base64>"}` |
| server → client | `{"type":"pronunciation_note"\|"grammar_note","text":"…"}` |
| server → client | `{"type":"done"}` / `{"type":"error","message":"…"}` |

## Scalability & web path

**Why WebSocket, not the HTTP endpoint.** API Gateway HTTP API + Lambda proxy
*buffers* the entire response — it cannot stream. The WebSocket API
(`StreamApi` / `StreamHandler`) lets one `converse` message fan out into many
pushed messages (`ApiGatewayManagementApi.post_to_connection`). It's fully
serverless and scales per-connection with no servers to manage, which is why it's
the path a real web app should use here rather than the buffered HTTP route (kept
only for simple request/response clients).

**Running in the browser (eventual web client).** The WebSocket protocol above is
transport-agnostic — a browser client uses the same messages. The pieces a web
client needs, none of which change the backend contract:

- **Mic + VAD in-browser:** `getUserMedia` + a WASM VAD (e.g. Silero via
  `@ricky0123/vad-web`) for the same no-Enter endpointing.
- **Live STT:** `@aws-sdk/client-transcribe-streaming` from the browser, with
  temporary credentials from a **Cognito Identity Pool** (add to the stack when the
  web client is built). Transcribe scales independently and keeps STT off your
  backend.
- **Playback:** decode the streamed MP3 chunks with the Web Audio API and queue
  them for gapless sentence-by-sentence playback.
- **Hosting:** ship the static app from **S3 + CloudFront**.

**Alternative — Amazon Nova Sonic.** For a purely conversational voice app, the
"what a real app uses today" answer is a speech-to-speech model like Nova Sonic
(Bedrock bidirectional streaming): it does STT, LLM, TTS, turn-detection and
barge-in in one stream. It does **not** do phoneme-level pronunciation scoring,
which is this project's differentiator — so the Wav2Vec2 branch would remain either
way. A Nova Sonic design typically needs a persistent connection host (Fargate/ECS)
to hold the bidirectional stream, rather than per-message Lambda. Worth adopting if
natural real-time conversation matters more than the serverless simplicity above.

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
