"use strict";

// ============================================================================
// Language Tutor web client. Two independent modes share one page:
//   - "sonic":  live speech-to-speech over a WebSocket (/ws) to the Fargate bridge.
//   - "custom": record one utterance, POST it to the HTTP API (/upload-url, /converse),
//               get a spoken reply plus pronunciation/grammar notes.
// The two never run at once; switching modes tears the other one down.
// ============================================================================

const IN_RATE = 16000;      // we always capture/send PCM16 mono 16kHz
const OUT_RATE = 24000;     // Nova Sonic sends PCM16 mono 24kHz
const FRAME_SAMPLES = 512;  // ~32ms per frame, the documented Sonic audioInput cadence

const $ = (id) => document.getElementById(id);
const els = {
  modeSonic: $("modeSonic"), modeCustom: $("modeCustom"),
  sonicControls: $("sonicControls"), customControls: $("customControls"),
  url: $("url"), token: $("token"), go: $("go"), end: $("end"), reset: $("reset"),
  record: $("record"), clearChat: $("clearChat"),
  dot: $("dot"), statusText: $("statusText"),
  log: $("log"), notes: $("notes"), notesPane: $("notesPane"),
  main: document.querySelector("main"), hint: $("hint"), sonicHint: $("sonicHint"),
};

// The custom-bot API is same-origin (CloudFront proxies /converse + /upload-url to the
// HTTP API), so relative paths just work. config.json only needs to supply the Sonic
// WebSocket URL, which lives on a different origin behavior.
const API_BASE = "";

// ---- shared state -----------------------------------------------------------
const state = {
  mode: "sonic",
  // audio
  mic: null, micCtx: null, playCtx: null, worklet: null,
  playCursor: 0, sources: new Set(),
  // sonic
  ws: null, userStopped: false, retries: 0, generation: 0,
  history: [],           // [{role:"USER"|"ASSISTANT", text}] — replayed on reconnect
  botSpec: "", botFinal: "", botBubble: null,
  framesSent: 0, deviceRate: null, micWatchdog: null,
  // custom
  recording: false, pcmChunks: [], busy: false,
};

function setStatus(text, kind) {
  els.statusText.textContent = text;
  els.dot.className = "dot" + (kind ? " " + kind : "");
}

function clearPlaceholder(node) {
  const ph = node.querySelector(".empty");
  if (ph) ph.remove();
}

// ---- endpoint / config ------------------------------------------------------
function defaultWsUrl() {
  if (!location.host || location.protocol === "file:") return "";
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

const savedUrl = localStorage.getItem("tutor.url") || "";
els.url.value = savedUrl || defaultWsUrl();
els.token.value = sessionStorage.getItem("tutor.token") || "";
if (savedUrl && savedUrl !== defaultWsUrl()) $("advanced").open = true;

async function loadConfig() {
  try {
    const resp = await fetch("config.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error("no config");
    const cfg = await resp.json();
    if (!savedUrl && typeof cfg.wsUrl === "string" && cfg.wsUrl) {
      els.url.value = cfg.wsUrl;
    }
    if (cfg.defaultMode === "custom") setMode("custom");
  } catch (_) {
    // Served locally with no config.json — the origin-derived ws:// URL is fine.
  }
}
loadConfig();

els.url.addEventListener("change", () => {
  const value = els.url.value.trim();
  if (!value || value === defaultWsUrl()) localStorage.removeItem("tutor.url");
  else localStorage.setItem("tutor.url", value);
});
els.reset.addEventListener("click", () => {
  els.url.value = defaultWsUrl();
  localStorage.removeItem("tutor.url");
});

// ---- mode switching ---------------------------------------------------------
function setMode(mode) {
  if (mode === state.mode && els.modeSonic.getAttribute("data-init")) return;
  stopEverything();
  state.mode = mode;
  els.modeSonic.setAttribute("data-init", "1");

  const sonic = mode === "sonic";
  els.modeSonic.setAttribute("aria-selected", String(sonic));
  els.modeCustom.setAttribute("aria-selected", String(!sonic));
  els.sonicControls.hidden = !sonic;
  els.customControls.hidden = sonic;
  els.notesPane.hidden = sonic;
  els.main.classList.toggle("single", sonic);

  // Reset panes and conversation memory: a transcript from one mode is not history for
  // the other (different models, different contract).
  state.history = [];
  els.log.replaceChildren();
  els.notes.replaceChildren();
  if (sonic) {
    els.log.innerHTML = '<p class="empty">Press <strong>Start talking</strong> and just '
      + 'speak Spanish. The tutor replies out loud, waits when you pause, and lets you '
      + 'talk over it.</p>';
    els.hint.textContent = "Headphones recommended — without them the mic hears the tutor "
      + "and may interrupt it.";
  } else {
    els.log.innerHTML = '<p class="empty">Press <strong>Record</strong>, say one thing in '
      + 'Spanish, then press <strong>Stop</strong>. You get a spoken reply plus '
      + 'pronunciation and grammar notes.</p>';
    els.notes.innerHTML = '<p class="empty">Pronunciation and grammar notes appear here '
      + 'after each utterance. An empty pane means nothing needed fixing.</p>';
    els.hint.textContent = "One utterance at a time — record, stop, listen, repeat.";
  }
  setStatus("Idle");
}
els.modeSonic.addEventListener("click", () => setMode("sonic"));
els.modeCustom.addEventListener("click", () => setMode("custom"));

// ---- conversation rendering -------------------------------------------------
function addTurn(role, text, live) {
  clearPlaceholder(els.log);
  const wrap = document.createElement("div");
  wrap.className = "turn " + (role === "USER" ? "user" : "bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (live ? " live" : "");
  bubble.textContent = text;
  wrap.appendChild(bubble);
  els.log.appendChild(wrap);
  els.log.scrollTop = els.log.scrollHeight;
  return bubble;
}

function addNote(category, quote, text) {
  clearPlaceholder(els.notes);
  const node = document.createElement("div");
  node.className = "note";
  const chip = document.createElement("span");
  chip.className = "chip " + category;
  chip.textContent = category;
  const q = document.createElement("div");
  q.className = "quote";
  q.textContent = "“" + quote + "”";
  const body = document.createElement("div");
  body.textContent = text;
  node.append(chip, q, body);
  els.notes.appendChild(node);
  els.notes.scrollTop = els.notes.scrollHeight;
}

// ---- audio: playback (sonic) ------------------------------------------------
function playChunk(b64) {
  const ctx = state.playCtx;
  if (!ctx) return;
  const raw = atob(b64);
  const pcm = new Int16Array(raw.length / 2);
  for (let i = 0; i < pcm.length; i++) {
    pcm[i] = (raw.charCodeAt(i * 2) | (raw.charCodeAt(i * 2 + 1) << 8)) << 16 >> 16;
  }
  const buf = ctx.createBuffer(1, pcm.length, OUT_RATE);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const startAt = Math.max(ctx.currentTime, state.playCursor);
  src.start(startAt);
  state.playCursor = startAt + buf.duration;
  state.sources.add(src);
  src.onended = () => state.sources.delete(src);
}

function flushPlayback() {
  for (const src of state.sources) { try { src.stop(); } catch (_) {} }
  state.sources.clear();
  state.playCursor = state.playCtx ? state.playCtx.currentTime : 0;
}

// ---- audio: capture ---------------------------------------------------------
function toBase64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let s = "";
  for (let i = 0; i < bytes.length; i += 4096) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 4096));
  }
  return btoa(s);
}

// Start the mic + capture worklet. `onFrame` receives each Int16Array PCM16@16k frame.
async function startMic(onFrame, withPlayback) {
  state.mic = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1, sampleRate: IN_RATE,
      echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    },
  });
  state.micCtx = new AudioContext({ sampleRate: IN_RATE });
  const ctxs = [state.micCtx];
  if (withPlayback) { state.playCtx = new AudioContext({ sampleRate: OUT_RATE }); ctxs.push(state.playCtx); }
  await Promise.all(ctxs.map((c) => c.resume().catch(() => {})));
  for (const c of ctxs) {
    if (c.state !== "running") {
      throw new Error(`The browser suspended an audio context (state: ${c.state}). Try again.`);
    }
  }
  if (state.playCtx) state.playCursor = state.playCtx.currentTime;

  await state.micCtx.audioWorklet.addModule("capture-worklet.js");
  const deviceRate = state.micCtx.sampleRate;
  state.deviceRate = deviceRate;
  const node = new AudioWorkletNode(state.micCtx, "capture", {
    processorOptions: { frameSamples: FRAME_SAMPLES, inputRate: deviceRate, outputRate: IN_RATE },
  });
  state.framesSent = 0;
  node.port.onmessage = (e) => {
    if (e.data && e.data.type === "ready") return;
    state.framesSent += 1;
    onFrame(e.data);
  };
  state.micCtx.createMediaStreamSource(state.mic).connect(node);
  node.connect(state.micCtx.createGain()).connect(state.micCtx.destination); // keep graph pulling
  state.worklet = node;
}

function stopMic() {
  flushPlayback();
  clearTimeout(state.micWatchdog);
  state.micWatchdog = null;
  if (state.worklet) { state.worklet.port.onmessage = null; state.worklet.disconnect(); }
  if (state.mic) state.mic.getTracks().forEach((t) => t.stop());
  for (const ctx of [state.micCtx, state.playCtx]) {
    if (ctx && ctx.state !== "closed") ctx.close();
  }
  state.worklet = state.mic = state.micCtx = state.playCtx = null;
}

// ============================================================================
// MODE A — Raw Nova Sonic (live WebSocket)
// ============================================================================
function wsUrl() {
  const base = els.url.value.trim().replace(/\/$/, "");
  if (!base) throw new Error("Enter the bridge WebSocket URL (stack output WsUrl).");
  if (location.protocol === "https:" && base.startsWith("ws://")) {
    throw new Error("This https page cannot use a ws:// endpoint. Use the wss:// URL "
      + "(WsUrl output), or open the client over http on localhost.");
  }
  const token = els.token.value.trim();
  return token ? base + "?token=" + encodeURIComponent(token) : base;
}

function onUserTranscript(text) {
  if (!text.trim()) return;
  closeBotTurn();
  addTurn("USER", text, false);
  state.history.push({ role: "USER", text });
}

function onBotTranscript(text, stage) {
  if (!text) return;
  if (stage === "FINAL") state.botFinal += text;
  else state.botSpec += text;
  const shown = state.botFinal.length >= state.botSpec.length ? state.botFinal : state.botSpec;
  if (!state.botBubble) state.botBubble = addTurn("ASSISTANT", shown, true);
  state.botBubble.textContent = shown;
  els.log.scrollTop = els.log.scrollHeight;
}

function closeBotTurn() {
  const text = (state.botFinal.length >= state.botSpec.length ? state.botFinal : state.botSpec).trim();
  if (text) state.history.push({ role: "ASSISTANT", text });
  if (state.botBubble) state.botBubble.classList.remove("live");
  state.botSpec = state.botFinal = "";
  state.botBubble = null;
}

function detach(ws) {
  if (!ws) return;
  ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
  try {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
    if (ws.readyState !== WebSocket.CLOSED) ws.close();
  } catch (_) {}
}

function connect(resume) {
  detach(state.ws);
  const generation = ++state.generation;
  const current = () => generation === state.generation;
  const ws = new WebSocket(wsUrl());
  state.ws = ws;
  setStatus(resume ? "Reconnecting…" : "Connecting…", "busy");

  ws.onopen = () => {
    if (!current()) { detach(ws); return; }
    setStatus("Connected — waiting for the tutor…", "busy");
    ws.send(JSON.stringify({ type: "start", history: state.history.slice(-20) }));
  };

  ws.onmessage = (ev) => {
    if (!current()) return;
    let m;
    try { m = JSON.parse(ev.data); } catch (_) { return; }
    switch (m.type) {
      case "ready":
        state.retries = 0;
        setStatus("Listening — just talk", "live");
        break;
      case "transcript":
        if (m.role === "USER") onUserTranscript(m.text);
        else onBotTranscript(m.text, m.stage);
        break;
      case "audio": playChunk(m.data); break;
      case "interrupted": flushPlayback(); break;
      case "done": break;
      case "error": setStatus("Error: " + m.message, "err"); break;
    }
  };

  ws.onerror = () => { if (current()) setStatus("Connection error", "err"); };

  ws.onclose = () => {
    if (!current()) return;
    state.ws = null;
    if (state.userStopped) { setStatus("Idle"); return; }
    // Nova Sonic caps a connection at ~8 minutes; reconnect and replay history.
    if (state.retries < 5) {
      state.retries += 1;
      setTimeout(() => connect(true), 400 * state.retries);
    } else {
      setStatus("Disconnected", "err");
      stopSonic();
    }
  };
}

async function startSonic() {
  try {
    els.go.disabled = true;
    sessionStorage.setItem("tutor.token", els.token.value.trim());
    state.userStopped = false;
    state.retries = 0;
    wsUrl(); // validate before touching the mic
    await startMic((frame) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: "audio", data: toBase64(frame) }));
      }
    }, true);
    state.micWatchdog = setTimeout(() => {
      if (state.framesSent === 0) {
        setStatus("Connected, but the mic is producing no audio — check the input device "
          + "and OS permissions", "err");
      }
    }, 2500);
    connect(false);
    els.end.disabled = false;
  } catch (err) {
    setStatus(err.message || String(err), "err");
    els.go.disabled = false;
    stopMic();
  }
}

function stopSonic() {
  state.userStopped = true;
  state.generation += 1;
  detach(state.ws);
  state.ws = null;
  stopMic();
  closeBotTurn();
  els.go.disabled = false;
  els.end.disabled = true;
  setStatus("Idle");
}

els.go.addEventListener("click", startSonic);
els.end.addEventListener("click", stopSonic);

// ============================================================================
// MODE B — Custom bot (record → upload → converse)
// ============================================================================
function encodeWav(chunks) {
  let total = 0;
  for (const c of chunks) total += c.length;
  const pcm = new Int16Array(total);
  let off = 0;
  for (const c of chunks) { pcm.set(c, off); off += c.length; }

  const bytesPerSample = 2;
  const dataSize = pcm.length * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);         // PCM chunk size
  view.setUint16(20, 1, true);          // PCM format
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, IN_RATE, true);
  view.setUint32(28, IN_RATE * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);         // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

async function startRecording() {
  try {
    state.pcmChunks = [];
    await startMic((frame) => state.pcmChunks.push(frame.slice(0)), false);
    state.recording = true;
    els.record.textContent = "Stop";
    els.record.classList.add("recording");
    setStatus("Recording — press Stop when done", "live");
  } catch (err) {
    setStatus(err.message || String(err), "err");
    stopMic();
  }
}

async function stopRecording() {
  state.recording = false;
  els.record.classList.remove("recording");
  els.record.textContent = "Record";
  stopMic();

  const chunks = state.pcmChunks;
  state.pcmChunks = [];
  if (!chunks.length) { setStatus("No audio captured", "err"); return; }

  els.record.disabled = true;
  state.busy = true;
  setStatus("Transcribing and thinking…", "busy");
  try {
    const wav = encodeWav(chunks);

    // 1. presigned upload URL
    const up = await fetch(`${API_BASE}/upload-url`).then((r) => r.json());
    // 2. upload the WAV
    await fetch(up.upload_url, { method: "PUT", headers: { "Content-Type": "audio/wav" }, body: wav });
    // 3. converse
    const resp = await fetch(`${API_BASE}/converse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_key: up.key, history: state.history.slice(-20) }),
    });
    const result = await resp.json();
    if (!resp.ok || result.error === "no_speech") {
      setStatus(result.error === "no_speech" ? "No speech detected — try again" : ("Error: " + (result.error || resp.status)), "err");
      return;
    }

    const transcript = (result.transcript || "").trim();
    if (transcript) { addTurn("USER", transcript, false); state.history.push({ role: "USER", text: transcript }); }
    const reply = result.reply || "";
    if (reply) { addTurn("ASSISTANT", reply, false); state.history.push({ role: "ASSISTANT", text: reply }); }
    if (result.pronunciation_note) addNote("pronunciation", transcript, result.pronunciation_note);
    if (result.grammar_note) addNote("grammar", transcript, result.grammar_note);

    if (result.reply_audio_url) {
      const audio = new Audio(result.reply_audio_url);
      audio.play().catch(() => {});
    }
    setStatus("Ready — press Record to continue", "live");
  } catch (err) {
    setStatus("Error: " + (err.message || err), "err");
  } finally {
    state.busy = false;
    els.record.disabled = false;
  }
}

els.record.addEventListener("click", () => {
  if (state.busy) return;
  if (state.recording) stopRecording();
  else startRecording();
});
els.clearChat.addEventListener("click", () => {
  state.history = [];
  els.log.innerHTML = '<p class="empty">Press <strong>Record</strong>, say one thing in '
    + 'Spanish, then press <strong>Stop</strong>.</p>';
  els.notes.innerHTML = '<p class="empty">Pronunciation and grammar notes appear here '
    + 'after each utterance.</p>';
  setStatus("Idle");
});

// ---- teardown shared by mode switches / unload ------------------------------
function stopEverything() {
  if (state.mode === "sonic") stopSonic();
  else {
    state.recording = false;
    if (els.record) { els.record.classList.remove("recording"); els.record.textContent = "Record"; }
    stopMic();
  }
}
window.addEventListener("beforeunload", () => { state.userStopped = true; });
