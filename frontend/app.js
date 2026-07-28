"use strict";

// --- audio constants: fixed by the Nova Sonic stream contract -----------------
const IN_RATE = 16000;     // we send PCM16 mono 16kHz
const OUT_RATE = 24000;    // it sends PCM16 mono 24kHz
const FRAME_SAMPLES = 512; // ~32ms per audioInput event, the documented cadence

const $ = (id) => document.getElementById(id);
const els = {
  url: $("url"), token: $("token"), go: $("go"), end: $("end"), reset: $("reset"),
  language: $("language"),
  dot: $("dot"), statusText: $("statusText"), log: $("log"), notes: $("notes"),
};

// Endpoint resolution, in precedence order:
//   1. an explicit override typed under "Endpoint" (kept in localStorage);
//   2. config.json, written by the CDK stack at deploy time — the authoritative
//      answer, and the one that stays correct if the bridge ever moves to a different
//      host than the page;
//   3. this page's own origin, which is right for the default single-distribution
//      topology and is the only option when serving web/ locally.
//
// The scheme matters and is never guessed loosely: an https page MUST use wss://,
// because browsers block ws:// from a secure page as mixed content. That is exactly why
// /ws is routed through CloudFront instead of pointing at the plain-HTTP load balancer.
function defaultWsUrl() {
  if (!location.host || location.protocol === "file:") return "";
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

// Only an explicit edit is remembered. Serving the page from somewhere new (a fresh
// distribution, or localhost for development) must not be overridden by a stale value.
const savedUrl = localStorage.getItem("tutor.url") || "";
els.url.value = savedUrl || defaultWsUrl();
els.token.value = sessionStorage.getItem("tutor.token") || "";
if (savedUrl && savedUrl !== defaultWsUrl()) $("advanced").open = true;

// The stack also publishes which languages it shipped, so the selector reflects what
// the bridge can actually serve rather than a hardcoded list.
function setLanguages(languages, selected) {
  els.language.replaceChildren();
  for (const { id, label } of languages) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = label;
    els.language.appendChild(option);
  }
  const remembered = localStorage.getItem("tutor.language");
  const wanted = [remembered, selected].find((v) =>
    v && languages.some((l) => l.id === v));
  if (wanted) els.language.value = wanted;
  els.language.disabled = languages.length < 2;
}

async function loadConfig() {
  try {
    const resp = await fetch("config.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error("no config");
    const cfg = await resp.json();
    if (!savedUrl && typeof cfg.wsUrl === "string" && cfg.wsUrl) {
      els.url.value = cfg.wsUrl;        // an explicit override still wins
    }
    if (Array.isArray(cfg.languages) && cfg.languages.length) {
      setLanguages(cfg.languages, cfg.defaultLanguage);
      return;
    }
  } catch (_) {
    // No config.json when serving frontend/ locally; fall back to the origin-derived
    // URL and a placeholder language so the control is never empty.
  }
  setLanguages([{ id: "", label: "As deployed" }], "");
}
loadConfig();

els.language.addEventListener("change", () => {
  localStorage.setItem("tutor.language", els.language.value);
  // A language is chosen when the Nova Sonic session opens, so switching mid-call
  // means reconnecting. Do it seamlessly if we are already talking.
  if (state.ws) restartForLanguage();
});

async function restartForLanguage() {
  setStatus("Switching language…");
  // Drop the transcript: replaying an English conversation into a German tutor as
  // "history" would just confuse it.
  state.history = [];
  stopAll();
  await start();
}

els.url.addEventListener("change", () => {
  const value = els.url.value.trim();
  if (!value || value === defaultWsUrl()) localStorage.removeItem("tutor.url");
  else localStorage.setItem("tutor.url", value);
});

els.reset.addEventListener("click", () => {
  els.url.value = defaultWsUrl();
  localStorage.removeItem("tutor.url");
});

let lastMeta = "";

const state = {
  ws: null, mic: null, micCtx: null, playCtx: null, worklet: null,
  playCursor: 0, sources: new Set(),
  history: [],            // [{role, text}] — replayed to resume after the 8-min cap
  userStopped: false, retries: 0,
  botSpec: "", botFinal: "", botBubble: null,
  pending: new Map(),      // transcript -> {node, count}
  framesSent: 0, micWatchdog: null, deviceRate: null,
  botTurnId: null, botTurnDone: false, userBubbles: new Map(),
};

function setStatus(text, kind) {
  els.statusText.textContent = text;
  els.dot.className = "dot" + (kind ? " " + kind : "");
}

function clearPlaceholder(node) {
  const ph = node.querySelector(".empty");
  if (ph) ph.remove();
}

// --- transcript rendering ----------------------------------------------------
// Append ?debug=1 to the URL to label each bubble with the role, stage and turn id the
// server actually sent. Diagnosing "that is not what I said" is guesswork otherwise:
// this distinguishes a mislabelled message from the model genuinely transcribing the
// wrong words.
const DEBUG = new URLSearchParams(location.search).get("debug") === "1";

function addTurn(role, text, live, meta) {
  clearPlaceholder(els.log);
  const wrap = document.createElement("div");
  wrap.className = "turn " + (role === "USER" ? "user" : "bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (live ? " live" : "");
  bubble.textContent = text;
  if (meta) bubble.title = meta;
  if (DEBUG && meta) {
    const tag = document.createElement("div");
    tag.className = "debug-tag";
    tag.textContent = meta;
    bubble.appendChild(tag);
  }
  wrap.appendChild(bubble);
  els.log.appendChild(wrap);
  els.log.scrollTop = els.log.scrollHeight;
  return bubble;
}

// The user bubble is written from Nova Sonic's transcript so it appears instantly, then
// replaced if the coaching lane's independent ASR disagrees.
function correctUserTranscript(m) {
  const bubble = state.userBubbles.get(m.turnId);
  if (!bubble || !m.text) return;
  bubble.textContent = m.text;
  bubble.title = `corrected by Transcribe (${m.language || "unknown"}); `
               + `Nova Sonic heard: ${m.was}`;
  if (DEBUG) {
    const tag = document.createElement("div");
    tag.className = "debug-tag";
    tag.textContent = `corrected lang=${m.language} was="${m.was}"`;
    bubble.appendChild(tag);
  }
  // Keep the resume history truthful too.
  for (let i = state.history.length - 1; i >= 0; i--) {
    if (state.history[i].role === "USER") { state.history[i].text = m.text; break; }
  }
}

function onUserTranscript(text, turnId) {
  if (!text.trim()) return;
  // A new learner turn means the tutor's turn is over, whatever the server said.
  // Belt and braces: without a close here, one missed turn_end would make every
  // later reply pile into a single bubble.
  closeBotTurn();
  const bubble = addTurn("USER", text, false, lastMeta);
  if (turnId) state.userBubbles.set(turnId, bubble);
  state.history.push({ role: "USER", text });
}

// Nova Sonic sends a SPECULATIVE preview of what it plans to say, then a FINAL
// sentence-level transcript of what it actually said. Show the preview immediately so
// the text tracks the voice, then let FINAL become the record.
//
// Grouping is by turnId (the model's completionId), never by event order. The FINAL
// transcript can arrive after the turn has otherwise ended, and closing the bubble on a
// timing signal made that trailing text open a second bubble — the reply appeared twice.
function onBotTranscript(text, stage, turnId) {
  if (!text) return;
  if (turnId && turnId !== state.botTurnId) {
    closeBotTurn();                    // a genuinely new reply
    state.botTurnId = turnId;
  }
  if (stage === "FINAL") state.botFinal += text;
  else state.botSpec += text;
  // SPECULATIVE is the full reply the model plans to speak; FINAL is a sentence-level
  // transcript of what it actually said, and it arrives in pieces. Preferring FINAL as
  // soon as any of it lands truncated every reply to its first sentence, so show
  // whichever accumulation is more complete.
  const shown = state.botFinal.length >= state.botSpec.length
    ? state.botFinal : state.botSpec;
  if (!state.botBubble) {
    state.botBubble = addTurn("ASSISTANT", shown, !state.botTurnDone, lastMeta);
  }
  // textContent replaces children, so the debug tag is re-attached below.
  state.botBubble.textContent = shown;
  if (DEBUG && lastMeta) {
    const tag = document.createElement("div");
    tag.className = "debug-tag";
    tag.textContent = lastMeta;
    state.botBubble.appendChild(tag);
  }
  state.botBubble.classList.toggle("live", !state.botTurnDone);
  els.log.scrollTop = els.log.scrollHeight;
}

// The tutor stopped speaking. The bubble stays open on purpose: a trailing FINAL
// transcript for the same turnId still belongs in it. It is closed when the next turn
// starts, when the learner speaks, or when the session ends.
function markBotTurnDone(turnId) {
  if (turnId && state.botTurnId && turnId !== state.botTurnId) return;
  state.botTurnDone = true;
  if (state.botBubble) state.botBubble.classList.remove("live");
}

function closeBotTurn() {
  const text = (state.botFinal.length >= state.botSpec.length
    ? state.botFinal : state.botSpec).trim();
  if (text) state.history.push({ role: "ASSISTANT", text });
  if (state.botBubble) state.botBubble.classList.remove("live");
  state.botSpec = state.botFinal = "";
  state.botBubble = null;
  state.botTurnId = null;
  state.botTurnDone = false;
}

// --- coach pane --------------------------------------------------------------
function coachPending(turn) {
  clearPlaceholder(els.notes);
  const node = document.createElement("div");
  node.className = "note pending";
  node.textContent = "Reviewing “" + turn + "”…";
  els.notes.appendChild(node);
  els.notes.scrollTop = els.notes.scrollHeight;
  state.pending.set(turn, { node, count: 0 });
}

function coachNote(turn, category, text) {
  clearPlaceholder(els.notes);
  const entry = state.pending.get(turn);
  const node = document.createElement("div");
  node.className = "note";
  const chip = document.createElement("span");
  chip.className = "chip " + category;
  chip.textContent = category;
  const quote = document.createElement("div");
  quote.className = "quote";
  quote.textContent = "“" + turn + "”";
  const body = document.createElement("div");
  body.textContent = text;
  node.append(chip, quote, body);
  if (entry) {
    entry.count += 1;
    els.notes.insertBefore(node, entry.node);
  } else {
    els.notes.appendChild(node);
  }
  els.notes.scrollTop = els.notes.scrollHeight;
}

function coachDone(turn, count, diag) {
  const entry = state.pending.get(turn);
  if (!entry) return;
  state.pending.delete(turn);
  if (count > 0) { entry.node.remove(); return; }

  // "Nothing to fix" must only be said when pronunciation was actually reviewed.
  // Claiming a clean bill of health while the phoneme scorer was unavailable is worse
  // than saying nothing: the learner assumes their pronunciation passed.
  const notScored = diag && diag.phonemeSkip;
  entry.node.className = notScored ? "note" : "note clean";
  entry.node.textContent = notScored
    ? "Grammar looked fine for “" + turn + "”. Pronunciation was not scored for this "
      + "turn, so it has not been checked."
    : "✓ “" + turn + "” — nothing to fix.";
  if (DEBUG && diag) {
    const tag = document.createElement("div");
    tag.className = "debug-tag";
    tag.textContent = JSON.stringify(diag);
    entry.node.appendChild(tag);
  }
}

// --- playback ---------------------------------------------------------------
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
  // Queue back-to-back so sentences play gapless even though chunks arrive faster
  // than real time.
  const startAt = Math.max(ctx.currentTime, state.playCursor);
  src.start(startAt);
  state.playCursor = startAt + buf.duration;
  state.sources.add(src);
  src.onended = () => state.sources.delete(src);
}

// Barge-in: Sonic stops generating, but audio already buffered here must be dropped
// or the tutor keeps talking over the user.
function flushPlayback() {
  for (const src of state.sources) { try { src.stop(); } catch (_) {} }
  state.sources.clear();
  state.playCursor = state.playCtx ? state.playCtx.currentTime : 0;
}

// --- capture ----------------------------------------------------------------

function toBase64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let s = "";
  for (let i = 0; i < bytes.length; i += 4096) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 4096));
  }
  return btoa(s);
}

async function startAudio() {
  state.mic = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1, sampleRate: IN_RATE,
      // The tutor plays through the speakers and the mic hears it. Without AEC that
      // bleed reads as the user barging in and cuts the tutor off constantly.
      echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    },
  });
  state.micCtx = new AudioContext({ sampleRate: IN_RATE });
  state.playCtx = new AudioContext({ sampleRate: OUT_RATE });

  // Both contexts are constructed AFTER awaiting the getUserMedia prompt, by which
  // point the click's user activation may have lapsed — so the browser's autoplay
  // policy can hand back a suspended context. A suspended context never runs the
  // capture worklet, so the socket opens and then absolutely nothing is sent. Resume
  // explicitly and refuse to continue if it stays suspended.
  await Promise.all([state.micCtx.resume(), state.playCtx.resume()].map((p) =>
    p.catch(() => {})));
  for (const [name, ctx] of [["microphone", state.micCtx], ["playback", state.playCtx]]) {
    if (ctx.state !== "running") {
      throw new Error(`The browser suspended the ${name} audio context `
                    + `(state: ${ctx.state}). Click Start talking again.`);
    }
  }
  state.playCursor = state.playCtx.currentTime;

  // Served from the same origin as this script, so no Blob URL gymnastics.
  await state.micCtx.audioWorklet.addModule("capture-worklet.js");

  // The constructor argument above is only a REQUEST: the browser may hand back the
  // device's native rate (48000 is common). Send the real rate to the worklet so it can
  // resample to exactly 16kHz — otherwise we would label 48kHz audio as 16kHz and the
  // model would hear the speech at a third speed and mis-transcribe it.
  const deviceRate = state.micCtx.sampleRate;
  state.deviceRate = deviceRate;
  if (deviceRate !== IN_RATE) {
    console.warn(`microphone context is ${deviceRate}Hz, resampling to ${IN_RATE}Hz`);
  }
  const node = new AudioWorkletNode(state.micCtx, "capture", {
    processorOptions: {
      frameSamples: FRAME_SAMPLES, inputRate: deviceRate, outputRate: IN_RATE,
    },
  });
  node.port.onmessage = (e) => {
    if (e.data && e.data.type === "ready") {
      console.debug("capture worklet", e.data);
      return;
    }
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "audio", data: toBase64(e.data) }));
      state.framesSent += 1;
    }
  };
  state.micCtx.createMediaStreamSource(state.mic).connect(node);
  // The worklet writes nothing to its output, so this routes silence — it exists only
  // to keep the graph pulling, which is what invokes process().
  node.connect(state.micCtx.createGain()).connect(state.micCtx.destination);
  state.worklet = node;

  // If frames still aren't flowing shortly after start, say so rather than sitting
  // there looking connected. This is the difference between "no audio reached the
  // model" and "the model had nothing to say".
  clearTimeout(state.micWatchdog);
  state.micWatchdog = setTimeout(() => {
    if (state.framesSent === 0) {
      setStatus("Connected, but the mic is producing no audio — check the input device "
              + "and OS permissions", "err");
    }
  }, 2500);
}

function stopAudio() {
  flushPlayback();
  clearTimeout(state.micWatchdog);
  state.micWatchdog = null;
  state.framesSent = 0;
  if (state.worklet) { state.worklet.port.onmessage = null; state.worklet.disconnect(); }
  if (state.mic) state.mic.getTracks().forEach((t) => t.stop());
  for (const ctx of [state.micCtx, state.playCtx]) {
    if (ctx && ctx.state !== "closed") ctx.close();
  }
  state.worklet = state.mic = state.micCtx = state.playCtx = null;
}

// --- connection -------------------------------------------------------------
function wsUrl() {
  const base = els.url.value.trim().replace(/\/$/, "");
  if (!base) throw new Error("Enter the bridge WebSocket URL (stack output WsUrl).");
  // A secure page cannot open an insecure socket, and the browser's own error for this
  // is opaque ("The operation is insecure"). Say what is actually wrong.
  if (location.protocol === "https:" && base.startsWith("ws://")) {
    throw new Error("This page is https, so it cannot use a ws:// endpoint. Use the "
                  + "wss:// URL (the WsUrl stack output) or open the client over http "
                  + "on localhost.");
  }
  const token = els.token.value.trim();
  return token ? base + "?token=" + encodeURIComponent(token) : base;
}

function connect(resume) {
  const ws = new WebSocket(wsUrl());
  state.ws = ws;
  setStatus(resume ? "Reconnecting…" : "Connecting…");

  ws.onopen = () => {
    // Distinguish "socket is up" from "the tutor's stream is up". If this status stays
    // put, the bridge accepted the connection but Nova Sonic never opened — a server
    // side problem (model access, IAM, region), not a microphone one.
    setStatus("Connected — waiting for the tutor…");
    // Replaying history is what makes a conversation survive Nova Sonic's 8 minute
    // per-connection cap: the new stream starts with the old context.
    ws.send(JSON.stringify({
      type: "start",
      language: els.language.value,
      history: state.history.slice(-20),
    }));
  };

  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (_) { return; }
    switch (m.type) {
      case "ready": {
        state.retries = 0;
        const rate = state.deviceRate && state.deviceRate !== IN_RATE
          ? ` (mic ${state.deviceRate}Hz -> ${IN_RATE}Hz)` : "";
        setStatus("Listening — just talk" + rate, "live");
        break;
      }
      case "transcript":
        lastMeta = `role=${m.role} stage=${m.stage} turn=${String(m.turnId).slice(0, 8)}`;
        console.debug("transcript", m);
        if (m.role === "USER") onUserTranscript(m.text, m.turnId);
        else onBotTranscript(m.text, m.stage, m.turnId);
        break;
      case "audio": playChunk(m.data); break;
      case "interrupted":
        // Drop queued audio, but do NOT close the bubble. Closing it cleared the turn
        // id, and the interrupted reply's trailing FINAL transcript then opened a
        // second bubble — the reply appeared twice on every barge-in.
        flushPlayback();
        markBotTurnDone(state.botTurnId);
        break;
      case "turn_end": markBotTurnDone(m.turnId); break;
      case "transcript_correction":
        // The independent ASR read the audio differently. Show what it heard: leaving
        // words on screen the learner never said is worse than a late correction.
        console.debug("transcript_correction", m);
        correctUserTranscript(m);
        break;
      case "coaching": coachPending(m.turn); break;
      case "feedback": coachNote(m.turn, m.category, m.text); break;
      case "coach_done":
        console.debug("coach_done", m);
        coachDone(m.turn, m.notes || 0, m.diag);
        break;
      case "error": setStatus("Error: " + m.message, "err"); break;
    }
  };

  ws.onerror = () => setStatus("Connection error", "err");

  ws.onclose = () => {
    state.ws = null;
    if (state.userStopped) { setStatus("Idle"); return; }
    // Nova Sonic caps a connection at 8 minutes; a close mid-conversation is
    // expected, not exceptional. Reconnect and carry the transcript over.
    if (state.retries < 5) {
      state.retries += 1;
      setStatus("Reconnecting…");
      setTimeout(() => connect(true), 400 * state.retries);
    } else {
      setStatus("Disconnected", "err");
      stopAll();
    }
  };
}

async function start() {
  try {
    els.go.disabled = true;
    sessionStorage.setItem("tutor.token", els.token.value.trim());
    state.userStopped = false;
    state.retries = 0;
    wsUrl();                 // validate before asking for the mic
    await startAudio();
    connect(false);
    els.end.disabled = false;
  } catch (err) {
    setStatus(err.message || String(err), "err");
    els.go.disabled = false;
    stopAudio();
  }
}

function stopAll() {
  state.userStopped = true;
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "stop" }));
    state.ws.close();
  }
  stopAudio();
  closeBotTurn();
  els.go.disabled = false;
  els.end.disabled = true;
  setStatus("Idle");
}

els.go.addEventListener("click", start);
els.end.addEventListener("click", stopAll);
window.addEventListener("beforeunload", () => { state.userStopped = true; });
