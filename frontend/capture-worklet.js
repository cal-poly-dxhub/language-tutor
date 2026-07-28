/**
 * Microphone capture worklet: device rate in, exactly 16kHz PCM16 out.
 *
 * WHY RESAMPLING IS HERE
 * `new AudioContext({ sampleRate: 16000 })` is a *request*. Browsers and devices may
 * hand back their native rate instead — 48000 is common. If that happens and we ship the
 * samples anyway, Nova Sonic is told the audio is 16kHz while it is really 48kHz, so it
 * hears the speech at a third of the correct speed. The result is not silence, which
 * would be obvious, but plausible-looking nonsense: mishearings that drift toward
 * whichever language the system prompt establishes. Resampling here makes the declared
 * rate true no matter what the device does.
 *
 * The conversion is linear interpolation with a fractional read cursor carried across
 * render quanta, so there is no click at block boundaries. Linear is adequate here: the
 * input is speech, the ratio is usually an exact 3:1, and the model is far more tolerant
 * of interpolation error than of a wrong sample rate.
 *
 * Output frames are `frameSamples` long (512 = ~32ms at 16kHz), the cadence the
 * bidirectional API documents for audioInput events.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.frameSamples = opts.frameSamples || 512;
    this.inputRate = opts.inputRate || sampleRate;   // sampleRate: worklet global
    this.outputRate = opts.outputRate || 16000;
    this.step = this.inputRate / this.outputRate;    // input samples per output sample

    this.buf = new Int16Array(this.frameSamples);
    this.n = 0;
    this.cursor = 0;        // fractional read position within the current input block
    this.previous = 0;      // last sample of the previous block, for interpolation
    this.hasPrevious = false;

    this.port.postMessage({
      type: "ready",
      inputRate: this.inputRate,
      outputRate: this.outputRate,
      resampling: Math.abs(this.step - 1) > 1e-9,
    });
  }

  emit(sample) {
    const clamped = Math.max(-1, Math.min(1, sample));
    this.buf[this.n++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    if (this.n === this.buf.length) {
      // slice(0) copies: the buffer is reused for the next frame.
      this.port.postMessage(this.buf.slice(0));
      this.n = 0;
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    // Fast path: rates already match, no interpolation needed.
    if (this.step === 1) {
      for (let i = 0; i < channel.length; i++) this.emit(channel[i]);
      return true;
    }

    // `cursor` is relative to the start of this block and may be negative, meaning the
    // sample to read straddles the boundary with the previous block.
    while (this.cursor < channel.length) {
      const base = Math.floor(this.cursor);
      const frac = this.cursor - base;
      const a = base < 0 ? (this.hasPrevious ? this.previous : channel[0]) : channel[base];
      const bIndex = base + 1;
      const b = bIndex < channel.length
        ? channel[bIndex]
        : (bIndex === channel.length ? channel[channel.length - 1] : a);
      this.emit(a + (b - a) * frac);
      this.cursor += this.step;
    }
    this.cursor -= channel.length;
    this.previous = channel[channel.length - 1];
    this.hasPrevious = true;
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
