# Language Tutor

Spoken Spanish practice with a real-time conversation partner, plus pronunciation and
grammar coaching that arrives *without interrupting the conversation*.

Two lanes, deliberately decoupled:

| | Fast lane — conversation | Slow lane — coaching |
|---|---|---|
| Model | Amazon Nova 2 Sonic (speech-to-speech) | Wav2Vec2 (phonemes) + Claude Haiku (judgement) |
| Latency | ~1s, streamed | 1–3s after you finish a sentence |
| Output | the tutor's voice | on-screen notes |
| Blocking? | never waits on the coach | never delays the conversation |

Nova Sonic does STT, reasoning, TTS, turn detection and barge-in inside one
bidirectional stream, so speaking feels like a phone call: no push-to-talk, no Enter
key, and you can talk over the tutor. It reports no phoneme detail, though — so the
project's original differentiator stays as an asynchronous side channel.

## Architecture

```
browser / CLI ──WebSocket──► Fargate bridge ──bidirectional stream──► Nova 2 Sonic
   mic (PCM16 16kHz)  ◄──── tutor audio (PCM16 24kHz) + transcripts ────┘
                                │                       │
                                │                       └─ tool: search_materials
                                │                            └─ Bedrock KB (course docs)
                                │
                    audio tee ──┴──► async coach  (per finished utterance, bounded)
                                        ├─ SageMaker Wav2Vec2 → expected vs actual phonemes
                                        └─ Claude Haiku → {"pronunciation":…, "grammar":…}
                                              └─► pushed to the client as coach notes
```

### Why a server and not Lambda

`InvokeModelWithBidirectionalStream` is a persistent HTTP/2 stream (8 minutes per
connection). API Gateway WebSocket + Lambda invokes once per message and cannot hold
one open, so the conversation runs on an always-on Fargate task behind an ALB. The
pre-Sonic serverless path is still in the tree — see [Legacy stack](#legacy-stack).

### Why coaching is not a Sonic tool call

Nova Sonic *can* call a tool, and an earlier design had it report corrections that way.
Teeing the audio instead is better on four counts:

- **Latency.** Sonic blocks waiting for a tool result, so grading would slow down the
  very turn it grades. The tee costs the conversation nothing.
- **Coverage.** The model decides when to call a tool. Every completed utterance gets
  reviewed here, deterministically.
- **Silence.** A tool result re-enters the conversation context, which tempts the tutor
  to say the correction out loud. The coach's output never touches Sonic.
- **Evidence.** A tool call carries text only. Phoneme scoring needs the raw audio,
  which never enters the model's context at all.

### Complete utterances only, never fragments

Nova 2 Sonic brackets the learner's speech with `userSpeechStart` / `userSpeechEnd`
events, and delivers the ASR transcript as one or more TEXT content blocks in between.
The bridge accumulates transcript text and dispatches to the coach when it has **both**
halves: `userSpeechEnd` (the learner stopped talking) and the transcript text. The two
arrive in either order, so the dispatch fires from whichever lands second, exactly once
per utterance — whether that utterance is one word or four sentences.

Content `stopReason` is deliberately *not* used for this. Verified against the live
model: a complete utterance still closes its ASR block with `PARTIAL_TURN`, so gating on
`END_TURN` means the coach never runs at all. `completionEnd` acts as a safety net so
nothing accumulated is left unreviewed. `endpointingSensitivity` is set to `LOW` so Sonic
waits longer before declaring your turn over, which is what learners who pause
mid-sentence need.

Locked down by `tests/test_turn_boundary.py`.

### Debugging a silent session

A rejected bidirectional stream is silent in a way that is easy to misread: the SDK
reports the stream as open, then returns **no events and no exception** for as long as it
stays open — the real error (missing model access, a denied IAM action, no egress) only
surfaces at teardown, by which point the client has usually disconnected. Two things
exist because of that:

- `tools/sonic_probe.py` runs the bridge's exact event sequence from your machine and
  prints every raw event, forcing teardown at the end so the true error appears:

  ```bash
  python3.12 -m venv .probe && .probe/bin/pip install -r bridge/requirements.txt
  .probe/bin/python tools/sonic_probe.py                 # as deployed
  .probe/bin/python tools/sonic_probe.py --region us-east-1
  ```

- the bridge reports it: 25s of audio with zero events back logs an error and tells the
  client to check model access and the task role. `{"type":"ping"}` returns a `debug`
  message with audio frames sent, event types received, model, region and last error.

### One transcript, and why

The coach grades Nova Sonic's transcript. An independent Amazon Transcribe pass over the
same teed audio was built, run, and then removed — every job it was given turned out to be
either unnecessary or harmful, and it cost a second container in the task, a wildcard IAM
grant, and per-turn latency.

The reasoning that put it there was that Sonic's transcript is conditioned on the
conversation, so speech in the learner's own language can come back rendered in the target
language. That is real. But the fix has to preserve one property: **expected phonemes must
describe what the learner was trying to say.** Sonic's transcript does, because it infers
intent. Transcribe writes what it literally heard — a learner aiming at "tortillas" and
missing gets "tortelas" — and phonemising *that* makes expected equal actual, so the
mispronunciation cancels out and no note is possible. It also manufactured grammar errors
out of its own mishearings, reporting a wrong pronoun where the learner had used the right
one.

Its other two jobs failed too. Correcting the on-screen transcript replaced correct text
with worse text in every observed case, including rendering correctly-spoken German as
English. Gating the review on "do the two transcripts agree?" gave a mishearing the power
to silence a turn.

What remains is simpler and rests on the audio rather than on a second opinion:

- a pronunciation note requires phoneme evidence, and requires that evidence to clear
  `MIN_TRUSTWORTHY_ACCURACY` — a claim about which sound was wrong needs sound to back it;
- grammar rests on the transcript alone, so no audio-derived condition suppresses it.
  Making a low phoneme score silence the whole turn was tried and reverted: it removed
  grammar coaching from precisely the learners whose audio aligns worst;
- whether a turn was in the learner's own language, and therefore needs no note, is a
  judgement the reviewer makes. It is told to return nothing for those turns.

### Why not a queue

Each review gets a private snapshot — the audio bytes, the transcript and the turn id are
copied at dispatch and passed by value into a per-turn task — so concurrent utterances
cannot overwrite each other's data. SQS would add a return path problem (the note has to
reach one specific WebSocket on one specific task), latency on a lane budgeted at 1–3s,
and durability that is worthless here: a correction nobody sees until the conversation has
moved on is worse than none. What *is* needed is a bound, so at most
`MAX_CONCURRENT_COACHING` (default 2) reviews run at once and further turns are skipped
rather than queued forever.

A queue would earn its place if coaching became offline — end-of-session reports, or
analytics over past utterances. Then S3 for the audio and SQS for the jobs is the right
shape.

### When a note actually appears

Most sentences produce nothing, which is the point — but nothing in the code decides
that. Haiku sees every completed utterance and judges it: backchannels, questions asked
in English, and correct sentences all come back as `null`, and regional variation
(listed per language in `languages/*.json`) is never flagged. There is deliberately no
keyword pre-filter, because a stopword list is a worse judge of "is this Spanish?" than
the model is, and every false negative silently costs the learner feedback they should
have received.

The bridge enforces only what the model cannot know or shouldn't be trusted with:

- a transcript with no words in it is never sent anywhere;
- near-silent audio never reaches Wav2Vec2, which would otherwise invent phonemes
  nobody spoke;
- a pronunciation note is dropped if the audio was never scored — sound cannot be
  inferred from spelling, so if the GPU endpoint is off the coach reviews grammar only;
- the same note is not repeated for the rest of the session.

## Teaching another language

Nothing in `bridge/` or `prompts/` names a language. All of it lives in
`languages/<name>.json`: the language names, the Nova Sonic voice, the accent features
the coach must never flag, and one example of each note style. The prompt files are
templates with `{{PLACEHOLDER}}` slots filled at synth time.

```bash
cp languages/spanish.json languages/german.json   # translate the content
./deploy.sh -c language=german
```

Profiles ship for all seven languages Nova 2 Sonic speaks — English, French, German,
Hindi, Italian, Portuguese and Spanish — each with its own voice and its own list of
accent features the coach must never flag. A missing field, an unknown voice id or an
unfilled placeholder fails the build rather than quietly degrading the tutor.

`-c language=` only sets the *default*. Every language's prompts are shipped to the
bridge (gzipped into one environment variable, ~6KB), so the browser client has a
language selector and switching reconnects the session — no redeploy.

## Deploy

```bash
pip install -r requirements.txt
./deploy.sh
```

`deploy.sh` is a thin wrapper around `npx cdk`, and you want it rather than calling
`cdk` yourself. The stack builds two container images, and the CDK CLI resolves its
builder as `process.env.CDK_DOCKER ?? "docker"` — so on a Mac with the Docker CLI
installed but no Docker Desktop daemon, a bare `npx cdk deploy` synthesizes fine and
then fails on the assets with
`failed to connect to the docker API at unix:///…/docker.sock`. If you see `docker build`
in that error, cdk was run without the wrapper.

The wrapper sets `CDK_DOCKER=finch`, starts the Finch VM if needed, and also puts
`bin/docker` (a shim that forwards to finch) first on `PATH` so anything that shells out
to a literal `docker` still works. Every other argument is forwarded:

```bash
./deploy.sh -c phonemes=false      # deploy without the GPU endpoint
./deploy.sh -c language=french     # teach a different language
./deploy.sh synth                  # any cdk subcommand
./deploy.sh diff
CDK_DOCKER=docker ./deploy.sh      # force a runtime yourself
```

As a backstop, synth refuses to run at all when no builder is reachable and tells you
to use the wrapper, so a mistaken `npx cdk deploy` fails in seconds instead of minutes.
Set `CDK_SKIP_RUNTIME_CHECK=1` to bypass that check (CI with a remote builder).

Deploys to **us-west-2** by default. Nova Sonic is region-limited (us-east-1, us-west-2,
ap-northeast-1, eu-north-1 at the time of writing); the model, the Knowledge Base and
the coach all share one region so there are no cross-region calls. Override with
`LANGUAGE_TUTOR_REGION`.

| Context flag | Effect |
|---|---|
| `-c language=<name>` | Teach a different language — picks `languages/<name>.json`. Default `spanish`. |
| `-c phonemes=false` | Skip the Wav2Vec2 GPU endpoint. Coach reviews grammar only. |
| `-c certificateArn=<acm arn>` | TLS on the ALB, so the client can use `wss://` from an https page. |

> **First build is slow.** The Wav2Vec2 image installs PyTorch and bakes in a ~1.2 GB
> model, cross-built for `linux/amd64`, so on Apple Silicon expect roughly 10–20 minutes
> the first time (cached afterwards). `-c phonemes=false` skips that image entirely and
> only builds the small bridge container.

> **Cost.** The phoneme endpoint is a 24/7 `ml.g4dn.xlarge`, which dominates the bill;
> the Fargate task and NAT gateway are the next largest fixed items. Use
> `-c phonemes=false` while iterating, and check the AWS Pricing Calculator for
> current figures.

### Post-deploy: Knowledge Base

CloudFormation needs a pre-existing vector store; the console's Quick Create builds
one. Run the script with no arguments for the exact console steps, then:

```bash
python3 setup_kb.py <KB_ID>
```

This patches the sync Lambda's environment *and* the bridge task definition, then rolls
the service (open sessions reconnect).

### Canvas LMS sync (optional)

```bash
aws secretsmanager put-secret-value --secret-id language-tutor/canvas-api-token \
  --secret-string '{"base_url":"https://your-school.instructure.com","token":"...","course_id":"12345"}'
```

Sync runs every 6 hours. Force one:

```bash
aws lambda invoke --function-name <SyncFunctionName> /dev/null
```

There is no public `/sync` endpoint — the previous stack exposed an unauthenticated
one, which is not worth the blast radius for a scheduled job.

## Access control

The ALB is internet-facing, so the bridge requires a shared secret on the WebSocket
upgrade (`?token=…` or `Authorization: Bearer …`). CDK generates it; read it with:

```bash
aws secretsmanager get-secret-value \
  --secret-id language-tutor/bridge-access-token \
  --query SecretString --output text
```

Without the token the connection is rejected with 401. This is a single shared secret,
not per-user auth — it stops casual abuse of your Bedrock bill, and is the right place
to swap in Cognito if this ever faces real users.

## Run it

### Browser (recommended)

The client is hosted for you: `./deploy.sh` uploads `frontend/` to S3 and serves it through
CloudFront, invalidating the cache each time.

```bash
open "$(aws cloudformation describe-stacks --stack-name LanguageTutorStack \
  --region us-west-2 --query "Stacks[0].Outputs[?OutputKey=='AppUrl'].OutputValue" \
  --output text)"
```

Paste the access token (below) and press **Start talking**. That is the only field: the
client derives the WebSocket endpoint from its own origin (`wss://<this host>/ws`), so
there is no URL to copy. Left pane is the conversation; right pane fills in with coach
notes a few seconds behind.

Use headphones. Without them the mic hears the tutor and Sonic's barge-in detection
cuts it off (browser echo cancellation is requested but is not perfect).

### Why the WebSocket goes through CloudFront

Two browser rules apply at once, and each alone is escapable:

- `getUserMedia` (the mic) requires a **secure context** — `https://`, or the specially
  exempted `http://localhost`. A plain `http://` page on any other host gets no mic.
- `ws://` is blocked as mixed content from an `https://` page, and is not auto-upgraded
  the way an image would be.

So `ws://` from an `http://` page is genuinely fine — that is what the local flow below
relies on. But hosting the page anywhere other than localhost forces `https`, which then
forbids `ws://`. Routing `/ws` through the same distribution means CloudFront terminates
TLS with its own `*.cloudfront.net` certificate, so the page talks `wss://` to its own
origin: no ACM certificate, no custom domain, no mixed content. That behavior disables
caching and forwards all viewer headers, because a cached WebSocket upgrade is a broken
one.

The ALB is still reachable directly (`AlbWsUrl`) for debugging, and is still gated by the
access token.

### Local, without CloudFront

```bash
python3 -m http.server 8000 --directory frontend
open http://localhost:8000
```

Open **Endpoint** and paste `AlbWsUrl`, plus the token. Overriding is only needed here:
on localhost the page's own origin is not the bridge. This works because
`http://localhost` is a secure context for the mic while still being allowed to open a
`ws://` socket.

### CLI

```bash
pip install pyaudio websockets boto3
python3 speech_client.py
```

Reads `WsUrl` (the CloudFront `wss://` endpoint) and the token from the stack
automatically. Override with `TUTOR_WS_URL` / `TUTOR_TOKEN`.

Both clients replay the last 20 turns as conversation history when a connection drops,
so a session survives Nova Sonic's 8-minute per-connection cap.

## Client ↔ bridge protocol

Client → server:

| Message | Meaning |
|---|---|
| `{"type":"start","language":"spanish","history":[{"role":"USER"\|"ASSISTANT","text":"…"}]}` | begin; both fields optional (history replayed for resume, language defaults to the deployed one) |
| `{"type":"audio","data":"<base64 pcm16 16kHz mono>"}` | ~32ms mic frame |
| `{"type":"text","content":"…"}` | typed input instead of speech |
| `{"type":"stop"}` | end the session |

Server → client:

| Message | Meaning |
|---|---|
| `{"type":"ready"}` | stream open, start talking |
| `{"type":"transcript","role":…,"text":…,"stage":"FINAL"\|"SPECULATIVE"}` | live text |
| `{"type":"audio","data":"<base64 pcm16 24kHz mono>"}` | tutor speech to play |
| `{"type":"coaching","turn":"…"}` | slow lane started on that utterance |
| `{"type":"feedback","category":"pronunciation"\|"grammar","text":…,"turn":…}` | a note |
| `{"type":"coach_done","turn":…,"notes":n}` | resolved; `n` may be 0 |
| `{"type":"interrupted"}` | barge-in — flush queued audio |
| `{"type":"turn_end"}` / `{"type":"error","message":…}` | |

## Prompts

- `prompts/conversation.txt` — the spoken tutor. The load-bearing rule is *never
  correct out loud*; corrections belong to the coach.
- `prompts/coach.txt` — the async reviewer. Returns strict JSON, `null` for "nothing to
  say", and is told at length what *not* to flag.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests -q          # needs Python >= 3.12
```

- `tests/test_coach.py` — silence trimming, JSON parsing, dedupe, the rule that
  pronunciation is never claimed without phoneme evidence, that grammar survives a poor
  phoneme score, and that no language vocabulary has crept back into the code.
- `tests/test_turn_boundary.py` — `END_TURN` vs `PARTIAL_TURN` dispatch, audio-tee
  alignment and reset, barge-in, tool resolution, token auth, failure reporting.
- `tests/test_language.py` — every profile renders every prompt with no placeholder
  left, and bad profiles fail loudly.

Infrastructure: `./deploy.sh synth`.

## Project structure

```
app.py                       # CDK entry point
cdk.json                     # cdk app config
deploy.sh                    # cdk wrapper that picks a working container runtime
cdk/                         # CDK stack package
  stack.py                   #   bridge + phoneme endpoint + KB + Canvas sync
  language.py                #   language profile loading + prompt rendering
languages/
  spanish.json               # all Spanish-specific content
  french.json                # second profile, proves the templates are neutral
prompts/
  conversation.txt           # Nova Sonic system prompt template
  coach.txt                  # async coach prompt template
bridge/                      # Fargate container (Docker asset)
  server.py                  #   Nova Sonic bridge, audio tee, turn detection
  coach.py                   #   async pronunciation/grammar coach
  Dockerfile, requirements.txt
container/                   # SageMaker Wav2Vec2 phoneme model (Docker asset)
lambda/sync.py               # Canvas LMS sync
frontend/                    # browser speech-to-speech client
  index.html                 #   structure
  styles.css                 #   presentation
  app.js                     #   protocol, capture, playback, rendering
  capture-worklet.js         #   audio-thread mic capture
speech_client.py             # CLI speech-to-speech client
setup_kb.py                  # post-deploy KB wiring
tests/                       # pytest suite
```

The pre-Sonic pipeline (Transcribe → Wav2Vec2 → Claude → Polly over API Gateway
WebSocket + Lambda) has been removed from this branch — Nova Sonic replaces all four
services. It remains on the `dev` branch if you need it:
`git show dev:lambda/stream_handler.py`.
