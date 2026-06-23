# Pronunciation Checker

Spanish conversational bot with pronunciation feedback. Uses SageMaker (Wav2Vec2 phonemes), Bedrock Claude Haiku 4.5 (LLM evaluation), and Polly Neural (spoken responses).

## Architecture

```
Client (mic) → Transcribe Streaming (speech-to-text)
             → SageMaker Wav2Vec2 (phoneme extraction)
             → Bedrock Claude Haiku 4.5 (dialect-aware evaluation)
             → Polly Neural TTS (spoken feedback)
             → Client (speaker)
```

The bot has a natural Spanish conversation with you. Pronunciation feedback appears as text annotations — the spoken reply stays conversational. The LLM understands regional variants (seseo, yeísmo, rioplatense) and only flags errors that impede comprehension.

## Deploy

```bash
pip install -r requirements.txt
npx cdk bootstrap   # first time only
npx cdk deploy
```

## Conversational Client

```bash
pip install amazon-transcribe pyaudio boto3 pydub simpleaudio
python converse_client.py
```

You speak Spanish naturally. The bot replies in Spanish (audio + text). Pronunciation notes appear below as text.

## System Prompt

The bot's personality and evaluation behavior is in `prompts/system.txt`. Edit it to change how strict/lenient the evaluation is, or to change the conversation style.

## Cost

- SageMaker ml.g4dn.xlarge: ~$0.74/hr
- Bedrock Claude Haiku 4.5: ~$0.80/1M input, $4/1M output tokens
- Polly Neural: $16/1M characters
- Transcribe Streaming: $0.024/min
