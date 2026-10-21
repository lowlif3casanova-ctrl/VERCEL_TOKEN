"""
synth.py — procedural instruments and DSP for the Pineal Gateway binaural session.

Everything here is pure numpy/scipy, float64 internally, phase-continuous, and
click-free by construction.  The binaural carriers are kept perfectly centred,
equal-level and dry; every other layer (drum, bowls, drones, chant, noise) is
free to move, blur and breathe.

Design references (see docs/DESIGN.md for the full source list):
  * Oster 1973 / Licklider 1950 — binaural beats need carriers < ~1 kHz, strongest
    in the 200–500 Hz region, beat < ~30 Hz.
  * ASSR literature (Picton 2003) — 40 Hz entrainment is strongest with amplitude
    modulated / isochronic stimulation, so gamma is delivered isochronically.
  * Terwagne & Bush 2011, Inácio et al. 2006 — Tibetan bowl modes follow the
    thin-shell law f_n ∝ n(n²−1)/√(n²+1): 1 : 2.83 : 5.42 : 8.77 : 12.87.
  * Levin & Edgerton 1999 — Tuvan sygyt isolates harmonics 6–12 by formant.
  * Kellet — economical pink-noise IIR approximation.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 48000
TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------
# Automation lanes
# ----------------------------------------------------------------------------
class Lane:
    """Breakpoint automation with raised-cosine (smoothstep) segments.

    points: list of (time_seconds, value).  Between breakpoints the value follows
    0.5 − 0.5·cos(πx), which has zero slope at both ends, so a ramp never has a
    corner that could be heard as a click or a sudden change of beat rate.
    """

    def __init__(self, points):
        pts = sorted(points, key=lambda p: p[0])
        self.t = np.array([p[0] for p in pts], dtype=np.float64)
        self.v = np.array([p[1] for p in pts], dtype=np.float64)
        if len(self.t) == 0:
            raise ValueError("Lane needs at least one breakpoint")

    def at(self, t):
        t = np.asarray(t, dtype=np.float64)
        if len(self.t) == 1:
            return np.full_like(t, self.v[0])
        idx = np.searchsorted(self.t, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.t) - 2)
        t0, t1 = self.t[idx], self.t[idx + 1]
        v0, v1 = self.v[idx], self.v[idx + 1]
        x = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
        s = 0.5 - 0.5 * np.cos(np.pi * x)
        out = v0 + (v1 - v0) * s
        out = np.where(t < self.t[0], self.v[0], out)
        out = np.where(t >= self.t[-1], self.v[-1], out)
        return out


def gate_lane(segments, fade=8.0, floor=0.0):
    """Build a Lane that is `level` inside each (start, end, level) segment and
    `floor` outside, with raised-cosine fades of `fade` seconds at both edges."""
    pts = []
    for (s, e, lvl) in sorted(segments):
        pts += [(s, floor), (s + fade, lvl), (e - fade, lvl), (e, floor)]
    if not pts:
        pts = [(0.0, floor)]
    return Lane(pts)


# ----------------------------------------------------------------------------
# Oscillator primitives (block-wise, phase-continuous)
# ----------------------------------------------------------------------------
def phase_increment(freq_hz, sr=SR):
    """Instantaneous frequency array -> phase array (radians), zero-based."""
    return TWO_PI * np.cumsum(freq_hz) / sr


class PhaseAcc:
    """Carries oscillator phase across render blocks (no discontinuities)."""

    def __init__(self, n=1):
        self.phase = np.zeros(n, dtype=np.float64)

    def advance(self, freq_hz_block, sr=SR):
        """freq_hz_block: (n_osc, n_samples) or (n_samples,) -> phase array."""
        f = np.atleast_2d(freq_hz_block)
        ph = TWO_PI * np.cumsum(f, axis=1) / sr + self.phase[:, None]
        self.phase = np.mod(ph[:, -1], TWO_PI)
        return ph if freq_hz_block.ndim == 2 else ph[0]


# ----------------------------------------------------------------------------
# Noise
# ----------------------------------------------------------------------------
KELLET = [
    (0.99886, 0.0555179), (0.99332, 0.0750759), (0.96900, 0.1538520),
    (0.86650, 0.3104856), (0.55000, 0.5329522), (-0.7616, -0.0168980),
]
KELLET_RMS = 3.0433   # measured RMS of the Kellet sum for unit white input (fixed, so blocks match)


def pink_noise(n, rng):
    """Kellet's economical pink-noise filter bank (within ±0.5 dB of 1/f over
    the audio band).  Returns unit-RMS float64 (one-shot; see PinkStream for
    block-wise rendering with carried filter state)."""
    w = rng.standard_normal(n)
    out = np.zeros(n)
    for a, c in KELLET:
        out += signal.lfilter([c], [1.0, -a], w)
    out += 0.5362 * w
    out += 0.115926 * np.concatenate(([0.0], w[:-1]))
    return out / KELLET_RMS


class PinkStream:
    """Pink noise rendered block by block with the six one-pole states and the
    one-sample FIR history carried across blocks, so there is no seam."""

    def __init__(self, rng):
        self.rng = rng
        self.zi = [np.zeros(1) for _ in KELLET]
        self.last_w = 0.0

    def next(self, n):
        w = self.rng.standard_normal(n)
        out = np.zeros(n)
        for k, (a, c) in enumerate(KELLET):
            y, self.zi[k] = signal.lfilter([c], [1.0, -a], w, zi=self.zi[k])
            out += y
        out += 0.5362 * w
        out += 0.115926 * np.concatenate(([self.last_w], w[:-1]))
        self.last_w = w[-1]
        return out / KELLET_RMS


class BandNoiseStream:
    """Band-passed white noise with carried SOS state and a fixed gain
    (no per-block normalisation), for the breath pacer and the 'air' layer."""

    def __init__(self, rng, lo, hi, sr=SR, order=4):
        self.rng = rng
        self.sos = signal.butter(order, [lo, hi], "bandpass", fs=sr, output="sos")
        self.zi = signal.sosfilt_zi(self.sos) * 0.0
        # calibrate the steady-state RMS once so every block has the same gain
        probe = signal.sosfilt(self.sos, np.random.default_rng(1).standard_normal(sr * 2))
        self.gain = 1.0 / (np.std(probe[sr // 2:]) + 1e-12)

    def next(self, n):
        y, self.zi = signal.sosfilt(self.sos, self.rng.standard_normal(n), zi=self.zi)
        return y * self.gain


def brown_noise(n, rng, sr=SR):
    w = rng.standard_normal(n)
    b = signal.lfilter([1.0], [1.0, -0.9995], w)
    sos = signal.butter(2, 25.0, "hp", fs=sr, output="sos")
    b = signal.sosfilt(sos, b)
    return b / (np.std(b) + 1e-12)


def bandpass(x, lo, hi, sr=SR, order=2):
    sos = signal.butter(order, [lo, hi], "bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def lowpass(x, fc, sr=SR, order=2):
    sos = signal.butter(order, fc, "lp", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def highpass(x, fc, sr=SR, order=2):
    sos = signal.butter(order, fc, "hp", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


# ----------------------------------------------------------------------------
# Envelopes / windows
# ----------------------------------------------------------------------------
def raised_cosine_fade(n_in, n_out, n_total):
    env = np.ones(n_total)
    if n_in > 0:
        x = np.linspace(0, np.pi, n_in)
        env[:n_in] = 0.5 - 0.5 * np.cos(x)
    if n_out > 0:
        x = np.linspace(0, np.pi, n_out)
        env[n_total - n_out:] = 0.5 + 0.5 * np.cos(x)
    return env


def add_at(buf, start_sample, sig_stereo):
    """Add a (2, m) stereo event into buf (2, n) at start_sample, clipping tails."""
    n = buf.shape[1]
    m = sig_stereo.shape[1]
    s = int(start_sample)
    if s >= n or s + m <= 0:
        return
    a = max(s, 0)
    b = min(s + m, n)
    buf[:, a:b] += sig_stereo[:, a - s:b - s]


def pan(mono, p):
    """Equal-power pan, p in [-1, 1]."""
    theta = (p + 1.0) * np.pi / 4.0
    return np.stack([np.cos(theta) * mono, np.sin(theta) * mono])


# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------
def frame_drum_hit(sr=SR, vel=1.0, rng=None, pitch=72.0, decay=0.22, dur=0.7):
    """Single stroke of a large shamanic frame drum (Harner/Siberian style).

    Pitch glide 1.9×→1× over ~25 ms (membrane tension transient), two body modes
    (f and 1.59 f, the (0,1) and (1,1) membrane modes), skin-slap noise, fast
    raised-cosine attack so there is no click."""
    rng = rng or np.random.default_rng(0)
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = pitch * (1.0 + 0.9 * np.exp(-t / 0.025))
    ph = phase_increment(f, sr)
    body = np.sin(ph) * np.exp(-t / decay)
    body += 0.35 * np.sin(1.59 * ph + 0.3) * np.exp(-t / (decay * 0.55))
    skin = bandpass(rng.standard_normal(n), 180.0, 2600.0, sr) * np.exp(-t / 0.028)
    hit = body + 0.32 * skin
    hit *= 1.0 - np.exp(-t / 0.0015)
    hit *= raised_cosine_fade(0, int(0.05 * sr), n)
    return (vel * hit / (np.max(np.abs(hit)) + 1e-12)).astype(np.float64)


def render_drum_track(n_total, segments, sr=SR, rng=None, width=0.35):
    """segments: list of dicts {start, end, rate_hz, gain, humanize_ms, accent_every,
    callback: bool}.  Returns (2, n_total) float32 stereo bus (dry)."""
    rng = rng or np.random.default_rng(7)
    bus = np.zeros((2, n_total), dtype=np.float32)
    variants = [frame_drum_hit(sr, 1.0, np.random.default_rng(100 + i),
                               pitch=70.0 + 3.0 * i, decay=0.20 + 0.02 * i) for i in range(4)]
    for seg in segments:
        start, end = seg["start"], seg["end"]
        rate = seg["rate_hz"]
        gain = seg["gain"]
        hum = seg.get("humanize_ms", 6.0) / 1000.0
        accent_every = seg.get("accent_every", 0)
        t = start
        i = 0
        # ramp in/out of the drum section so it never starts abruptly
        ramp = seg.get("ramp_s", 12.0)
        while t < end:
            vel = 0.78 + 0.22 * rng.random()
            if accent_every and i % accent_every == 0:
                vel *= 1.18
            env = 1.0
            if t - start < ramp:
                env = 0.5 - 0.5 * np.cos(np.pi * (t - start) / ramp)
            if end - t < ramp:
                env *= 0.5 - 0.5 * np.cos(np.pi * (end - t) / ramp)
            v = variants[i % len(variants)]
            pos = (rng.random() * 2 - 1) * width
            add_at(bus, int((t + rng.normal(0, hum)) * sr), (pan(v, pos) * (vel * gain * env)).astype(np.float32))
            t += 1.0 / rate
            i += 1
    return bus


def harner_callback(n_total, start, sr=SR, rng=None, gain=1.0, fast_rate=8.0, fast_dur=30.0,
                    ground_rate=1.2, ground_dur=60.0):
    """Journey call-back signal.

    Foundation for Shamanic Studies practice (shamanism.org/drumming; Harner's
    1997 recording): the journey drum ends with ~30 s of markedly faster
    drumming (7-10 Hz) followed by a minute or two of slow re-grounding beats.
    The 'four groups of seven sharp strokes' that opens and closes the fast
    passage here is a widespread practitioner convention (not an FSS standard)
    and is included because many listeners already know it as 'the call-back'.
    Returns (2, n) stereo bus and the end time in seconds."""
    rng = rng or np.random.default_rng(11)
    bus = np.zeros((2, n_total), dtype=np.float32)
    hit = frame_drum_hit(sr, 1.0, np.random.default_rng(200), pitch=74.0, decay=0.17)
    slow_hit = frame_drum_hit(sr, 1.0, np.random.default_rng(201), pitch=66.0, decay=0.26, dur=0.9)
    t = start

    def seven(t):
        for k in range(7):
            add_at(bus, int(t * sr), (pan(hit, 0.0) * (gain * 1.05)).astype(np.float32))
            t += 0.28
        return t + 0.9

    for _ in range(4):
        t = seven(t)
    t += 0.5
    t_end_fast = t + fast_dur
    while t < t_end_fast:
        v = 0.7 + 0.3 * rng.random()
        add_at(bus, int((t + rng.normal(0, 0.004)) * sr), (pan(hit, 0.0) * (gain * v)).astype(np.float32))
        t += 1.0 / fast_rate
    t += 0.8
    for _ in range(4):
        t = seven(t)
    t += 1.5
    # slow re-grounding beats (FSS: 1-2 min at ~1-1.5 Hz), fading out
    t_end_ground = t + ground_dur
    t_g0 = t
    while t < t_end_ground:
        env = 1.0 if t - t_g0 < ground_dur * 0.6 else 0.5 + 0.5 * np.cos(np.pi * (t - t_g0 - ground_dur * 0.6) / (ground_dur * 0.4))
        add_at(bus, int((t + rng.normal(0, 0.006)) * sr), (pan(slow_hit, 0.0) * (gain * 0.9 * env)).astype(np.float32))
        t += 1.0 / ground_rate
    return bus, t


def pututu_cluster(sr=SR, f0=340.25, offsets=(0.0, 3.9, 7.83), dur=14.0, gain=1.0, rng=None):
    """Chavín-style conch (pututu) cluster: several Strombus trumpets whose
    fundamentals lie within a few Hz of each other (measured 272-340 Hz by
    Cook et al. 2010) produce strong slow beats when blown together (Kolar
    2014).  340.25 Hz (5:2 of Om) is the top of the measured range and sits
    one critical band from both the 272 and 408 Hz binaural carriers.
    Breathy, with a swell, placed as the 'upper-world' call."""
    rng = rng or np.random.default_rng(13)
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for off in offsets:
        f = f0 + off
        ph = TWO_PI * f * t + rng.uniform(0, TWO_PI)
        tone = np.sin(ph) + 0.45 * np.sin(2.05 * ph + 0.4) + 0.18 * np.sin(3.1 * ph + 1.1)
        tone *= 1.0 + 0.012 * np.sin(TWO_PI * 5.3 * t + rng.uniform(0, TWO_PI))   # lip vibrato
        out += tone
    breath = bandpass(rng.standard_normal(n), 900.0, 3500.0, sr, order=2)
    out += 0.08 * breath / (np.std(breath) + 1e-9)
    env = raised_cosine_fade(int(0.3 * n), int(0.45 * n), n)
    out *= env
    return gain * out / (np.max(np.abs(out)) + 1e-12)


def chamber_drone_block(acc, t_block, f0=102.08, sr=SR, n_harm=9):
    """Male-register 'stone chamber' drone (Jahn, Devereux & Ibison 1996 found
    95-120 Hz resonances in Neolithic chambers; the Ħal Saflieni Hypogeum
    resonates near 114 Hz).  102.08 Hz is the pure 4:3 below the 136.1 Hz Om
    (a fifth above the 68 Hz didgeridoo), inside the measured range and the
    tonal system.  Only harmonics 1, 5 and 7 sound (hollow, clarinet-like):
    the even harmonics would land on the 204 Hz pulse carrier and the 408 Hz
    binaural carrier, and the 3rd within a critical band of 272 Hz."""
    freqs = f0 * np.arange(1, n_harm + 1)[:, None] * np.ones((1, len(t_block)))
    ph = acc.advance(freqs, sr)
    w = np.array([1.0, 0.0, 0.0, 0.0, 0.12, 0.0, 0.05, 0.0, 0.0])[:n_harm, None]
    slow = 1.0 + 0.15 * np.sin(TWO_PI * 0.05 * t_block + np.arange(n_harm)[:, None] * 0.9)
    return np.sum(w * slow * np.sin(ph), axis=0)


BOWL_RATIOS = np.array([1.0, 2.83, 5.42, 8.77, 12.87])          # thin-shell law
BOWL_AMPS = np.array([1.0, 0.55, 0.27, 0.12, 0.05])
BOWL_T60 = np.array([30.0, 18.0, 9.0, 4.5, 2.2])
# Bonshō (Japanese temple bell): measured Hōjōbō bell modes 137.8 / 301.8 / 378.6 Hz
BONSHO_RATIOS = np.array([1.0, 2.190, 2.747, 4.31, 6.02])
BONSHO_AMPS = np.array([1.0, 0.70, 0.45, 0.18, 0.08])
BONSHO_T60 = np.array([45.0, 22.0, 14.0, 6.0, 3.0])


def singing_bowl(f0, sr=SR, dur=38.0, gain=1.0, rng=None, strike=True, rim_sing=0.0,
                 ratios=None, amps=None, t60=None, split_hz=None):
    """Tibetan singing bowl (or, with the BONSHO_* tables, a temple bell).

    Modes follow the thin-shell law (Terwagne & Bush 2011): 1 : 2.83 : 5.42 :
    8.77 : 12.87.  Each mode is a doublet; real bowls beat at ~2-10 Hz (Inácio,
    Henrique & Antunes 2006).  `split_hz` sets the fundamental's doublet split
    directly, so a bowl can be made to 'wah' at the session's beat rate and
    carry it acoustically (monaurally).  Higher modes decay faster.  Optional
    rim-singing sustain (friction excitation) adds a slow swell."""
    rng = rng or np.random.default_rng(3)
    n = int(dur * sr)
    t = np.arange(n) / sr
    ratios = BOWL_RATIOS if ratios is None else np.asarray(ratios)
    amps = BOWL_AMPS if amps is None else np.asarray(amps)
    t60 = (BOWL_T60 if t60 is None else np.asarray(t60)) * (dur / 38.0)
    out = np.zeros(n)
    for k, (r, a, tt) in enumerate(zip(ratios, amps, t60)):
        f = f0 * r
        if f > 0.45 * sr:
            continue
        if split_hz is not None and k == 0:
            split = float(split_hz)
        else:
            split = f * rng.uniform(0.004, 0.012)   # doublet split -> ~1-6 Hz beating
        decay = np.exp(-6.9078 * t / tt)
        ph1 = TWO_PI * (f - split / 2) * t + rng.uniform(0, TWO_PI)
        ph2 = TWO_PI * (f + split / 2) * t + rng.uniform(0, TWO_PI)
        out += a * decay * (np.sin(ph1) + 0.9 * np.sin(ph2)) * 0.5
    if strike:
        ns = int(0.006 * sr)
        burst = bandpass(rng.standard_normal(ns), 800.0, 6000.0, sr) * np.exp(-np.arange(ns) / (0.0015 * sr))
        out[:ns] += 0.25 * burst
        out *= 1.0 - np.exp(-t / 0.003)
    if rim_sing > 0:
        # rim-singing sustain on the SAME doublet as the fundamental, so the bowl keeps
        # beating at split_hz (a single un-split tone would beat against the doublet at split/2)
        swell = (0.5 - 0.5 * np.cos(np.pi * np.clip(t / 6.0, 0, 1)))
        split0 = float(split_hz) if split_hz is not None else f0 * 0.008
        rim = 0.5 * (np.sin(TWO_PI * (f0 - split0 / 2) * t) + 0.9 * np.sin(TWO_PI * (f0 + split0 / 2) * t + 0.4))
        out += rim_sing * swell * np.exp(-t / (dur * 0.6)) * rim * (0.6 + 0.4 * np.sin(TWO_PI * 0.37 * t))
    out *= raised_cosine_fade(0, int(2.0 * sr), n)
    return gain * out / (np.max(np.abs(out)) + 1e-12)


def tingsha(sr=SR, f0=2960.0, detune_hz=9.0, dur=5.0, gain=1.0, rng=None):
    """Tibetan tingsha: a pair of small bronze cymbals, >2 kHz, deliberately
    slightly mis-tuned so the pair beats (here ~9 Hz).  Short bright decay."""
    rng = rng or np.random.default_rng(4)
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f, a, tt in ((f0, 1.0, 3.2), (f0 + detune_hz, 0.9, 3.0), (f0 * 2.71, 0.25, 1.2), (f0 * 2.71 + 14, 0.22, 1.1)):
        if f < 0.45 * sr:
            out += a * np.exp(-6.9078 * t / tt) * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    ns = int(0.004 * sr)
    out[:ns] += 0.3 * bandpass(rng.standard_normal(ns), 2000.0, 9000.0, sr) * np.exp(-np.arange(ns) / (0.001 * sr))
    out *= 1.0 - np.exp(-t / 0.0015)
    out *= raised_cosine_fade(0, int(1.0 * sr), n)
    return gain * out / (np.max(np.abs(out)) + 1e-12)


def dungchen_swell(sr=SR, f0=68.05, dur=28.0, gain=1.0, rng=None, slide=-0.012):
    """Dungchen-like long-horn swell: a 15-30 s crescendo/decrescendo on a low
    pedal note with a slight upward pitch slide into the note (brass-like
    harmonic series, 1/h^0.9), then a slow release."""
    rng = rng or np.random.default_rng(6)
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 * (1.0 + slide * np.exp(-t / 2.5))
    ph = phase_increment(f, sr)
    out = np.zeros(n)
    for h in range(1, 16):
        out += np.sin(h * ph + rng.uniform(0, TWO_PI)) / (h ** 0.9) * (1.0 if h < 9 else 0.6)
    env = raised_cosine_fade(int(0.45 * n), int(0.5 * n), n)
    out *= env
    return gain * out / (np.max(np.abs(out)) + 1e-12)


def thunder_swell(sr=SR, dur=24.0, gain=1.0, rng=None, lp_hz=180.0):
    """Distant thunder: band-limited brown noise with a slow irregular swell
    (for the Bardo 'thousand thunders' episode, kept distant and soft)."""
    rng = rng or np.random.default_rng(8)
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = brown_noise(n, rng, sr)
    x = lowpass(x, lp_hz, sr, order=2)
    lf = lowpass(rng.standard_normal(n), 0.9, sr, order=1)
    lf = lf / (np.std(lf[sr:]) + 1e-12)          # normalise by its own statistics
    rumble = np.clip(0.6 + 0.4 * lf, 0.0, 1.5)
    env = raised_cosine_fade(int(0.35 * n), int(0.55 * n), n)
    out = x * env * rumble
    return gain * out / (np.max(np.abs(out)) + 1e-12)


class AirStream:
    """Very quiet 3-8 kHz 'air' (the band Tomatis called 'charging'): a slowly
    breathing band of filtered noise, stereo decorrelated, rendered block-wise
    on the caller's absolute time axis with carried filter state."""

    def __init__(self, rng, sr=SR, lo=3000.0, hi=8000.0, am_hz=0.07):
        self.streams = [BandNoiseStream(np.random.default_rng(rng.integers(1 << 31)), lo, hi, sr, order=2) for _ in range(2)]
        self.am_hz = am_hz

    def next(self, t_block):
        n = len(t_block)
        out = np.zeros((2, n))
        for ch in range(2):
            out[ch] = self.streams[ch].next(n) * (0.6 + 0.4 * np.sin(TWO_PI * self.am_hz * t_block + ch * 1.7))
        return out


class AdditiveDrone:
    """Harmonic drone with time-varying per-harmonic weights (formant sweeps).

    Used for: Gyuto/Tuvan overtone chant, didgeridoo, tanpura 'Om' drone.
    Rendered block-wise with phase carry so a 50-minute drone has no seams.
    """

    def __init__(self, f0, n_harm=36, rolloff=0.85, sr=SR, notch=(), notch_gain=0.08):
        """notch: harmonic numbers to attenuate (× notch_gain).  Used to keep a
        drone's harmonics off the binaural carriers: a steady tone at exactly a
        carrier's frequency in both ears turns part of the binaural beat into a
        half-rate monaural beat (carrier f±b/2 against the drone's f)."""
        self.f0 = f0
        self.h = np.arange(1, n_harm + 1, dtype=np.float64)
        self.base = 1.0 / (self.h ** rolloff)
        for k in notch:
            if 1 <= k <= n_harm:
                self.base[k - 1] *= notch_gain
        self.acc = PhaseAcc(n_harm)
        self.sr = sr

    def render(self, t_block, f0_mult=None, formant_center=None, formant_width=1.6,
               formant_gain=5.0, vibrato_hz=0.0, vibrato_depth=0.0, extra_weights=None):
        """t_block: time axis of this block (s).  formant_center: array of the
        harmonic number the formant sits on (may vary over the block)."""
        n = len(t_block)
        f0 = np.full(n, self.f0) if f0_mult is None else self.f0 * f0_mult
        if vibrato_hz > 0:
            f0 = f0 * (1.0 + vibrato_depth * np.sin(TWO_PI * vibrato_hz * t_block))
        freqs = self.h[:, None] * f0[None, :]
        ph = self.acc.advance(freqs, self.sr)
        w = np.repeat(self.base[:, None], n, axis=1)
        if formant_center is not None:
            fc = np.asarray(formant_center)
            w = w * (1.0 + formant_gain * np.exp(-((self.h[:, None] - fc[None, :]) ** 2) / (2 * formant_width ** 2)))
        if extra_weights is not None:
            w = w * extra_weights
        # anti-alias: silence harmonics above 0.45 fs
        w = np.where(freqs < 0.45 * self.sr, w, 0.0)
        return np.sum(w * np.sin(ph), axis=0)


def vowel_formants(vowel):
    """Approximate adult male formant frequencies (Hz) and bandwidths for the
    sacred vowel set.  Values from Peterson & Barney 1952 / Hillenbrand 1995,
    adapted for chant (slightly lowered F1 for an open-throat tone)."""
    table = {
        "A":  ([700, 1150, 2500], [80, 100, 130]),   # ah  (Α / heart)
        "E":  ([520, 1800, 2550], [70, 100, 130]),   # eh  (Ε)
        "EH": ([450, 2000, 2650], [70, 100, 130]),   # ē   (Η, long e)
        "I":  ([290, 2250, 3000], [60, 100, 140]),   # ee  (Ι / crown)
        "O":  ([500,  850, 2450], [70,  90, 130]),   # oh  (Ο / solar plexus)
        "U":  ([310,  800, 2300], [60,  90, 130]),   # oo  (Υ / sacral)
        "OM": ([400,  750, 2400], [70,  90, 130]),   # aw→m (Ω / crown)
        "M":  ([250,  900, 2200], [90, 150, 200]),   # hum, closed lips
        "UH": ([600, 1200, 2450], [80, 100, 130]),   # uh (root)
    }
    return table[vowel]


def chant_voice(n, f0, vowel_sequence, sr=SR, rng=None, vibrato_hz=4.8, vibrato_depth=0.004,
                voices=3, detune=0.004, pan_spread=0.6, fade_s=1.5):
    """Formant-synthesised sacred vowel chant.

    vowel_sequence: list of (start_s, end_s, vowel_key) or (start_s, end_s,
    vowel_key, f0_hz) — the optional 4th element lets a ladder ascend pitch per
    syllable (the pitch glides over `fade_s`).  A pulse-rich glottal source
    (harmonic series with 1/h^0.6 rolloff and slow jitter) is passed through
    three 2-pole resonators whose centre frequencies cross-fade between vowels.
    `voices` detuned copies give a small choir.  Returns (2, n) stereo."""
    rng = rng or np.random.default_rng(5)
    t = np.arange(n) / sr
    out = np.zeros((2, n))
    # per-sample target formants and pitch
    F = np.zeros((3, n))
    B = np.zeros((3, n))
    P = np.zeros(n)
    env = np.zeros(n)
    nf = int(fade_s * sr)
    wsum = np.zeros(n)
    for item in vowel_sequence:
        s, e, vw = item[0], item[1], item[2]
        f_item = item[3] if len(item) > 3 else f0
        fr, bw = vowel_formants(vw)
        # each segment is widened by fade_s/2 on both sides so that adjacent
        # segments overlap by fade_s and their raised-cosine edges sum to 1:
        # a true amplitude-constant cross-fade with formant and pitch glides
        a = max(int((s - fade_s / 2) * sr), 0)
        b = min(int((e + fade_s / 2) * sr), n)
        if b <= a:
            continue
        seg = np.zeros(n)
        nfa = min(nf, (b - a) // 2)
        seg[a:b] = raised_cosine_fade(nfa, nfa, b - a)
        wsum += seg
        env = np.minimum(env + seg, 1.0)
        P += f_item * seg
        for k in range(3):
            F[k] += fr[k] * seg
            B[k] += bw[k] * seg
    # weighted averages across overlapping segments; neutral formant where silent
    neutral_f, neutral_b = vowel_formants("M")
    for k in range(3):
        F[k] = np.where(wsum > 1e-4, F[k] / np.maximum(wsum, 1e-4), neutral_f[k])
        B[k] = np.where(wsum > 1e-4, B[k] / np.maximum(wsum, 1e-4), neutral_b[k])
    pitch = np.where(wsum > 1e-4, P / np.maximum(wsum, 1e-4), f0)
    for v in range(voices):
        d = 1.0 + detune * (v - (voices - 1) / 2)
        lf = lowpass(rng.standard_normal(n), 3.0, sr)
        jitter = 1.0 + 0.0025 * lf / (np.std(lf[sr:]) + 1e-12)     # ~0.25 % slow pitch wander
        f = pitch * d * jitter * (1.0 + vibrato_depth * np.sin(TWO_PI * vibrato_hz * t + rng.uniform(0, TWO_PI)))
        ph = phase_increment(f, sr)
        src = np.zeros(n)
        fmax = float(np.max(pitch)) * 1.05
        for h in range(1, 40):
            if fmax * h > 0.45 * sr:
                break
            src += np.sin(h * ph + rng.uniform(0, TWO_PI)) / (h ** 0.6)
        src *= 1.0 / 6.0
        # time-varying formant resonators: process in 4096-sample blocks with
        # coefficients updated per block (sufficient for 1.5 s cross-fades)
        y = np.zeros(n)
        blk = 4096
        zi = [np.zeros(2) for _ in range(3)]
        for a in range(0, n, blk):
            b = min(a + blk, n)
            x = src[a:b]
            acc = np.zeros(b - a)
            for k in range(3):
                fc = float(np.mean(F[k, a:b]))
                bw = float(np.mean(B[k, a:b]))
                r = np.exp(-np.pi * bw / sr)
                theta = TWO_PI * fc / sr
                # unity gain at the resonator's own centre frequency
                b0 = (1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2.0 * theta) + r * r)
                acoef = [1.0, -2.0 * r * np.cos(theta), r * r]
                seg, zi[k] = signal.lfilter([b0], acoef, x, zi=zi[k])
                acc += seg * (1.0, 0.7, 0.4)[k]
            y[a:b] = acc
        # slow loudness equalisation: formant/harmonic alignment makes some vowels
        # 2x louder than others at a given pitch; level the syllables (0.5 s window)
        lvl = np.sqrt(lowpass(y * y, 1.5, sr, order=1).clip(0) + 1e-9)
        y = y / np.maximum(lvl, 0.02 * np.max(lvl)) * env
        p = pan_spread * (v - (voices - 1) / 2) / max(1, (voices - 1) / 2)
        out += pan(y, p)
    out /= (np.max(np.abs(out)) + 1e-12)
    return out


def synthetic_ir(sr=SR, t60=3.6, predelay=0.018, rng=None, hf_damp_hz=3800.0):
    """Stereo impulse response: decorrelated exponentially decaying noise with a
    darkening tail (temple / cave character)."""
    rng = rng or np.random.default_rng(9)
    n = int((t60 * 1.1 + predelay) * sr)
    t = np.arange(n) / sr
    env = np.exp(-6.9078 * np.maximum(t - predelay, 0) / t60)
    env[t < predelay] = 0.0
    ir = np.stack([rng.standard_normal(n), rng.standard_normal(n)]) * env
    ir = lowpass(ir, hf_damp_hz, sr, order=1)
    # early reflections a little brighter: light high shelf via mix
    ir = highpass(ir, 90.0, sr, order=1)
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True)) + 1e-12
    return ir


def convolve_reverb(bus, ir):
    """Overlap-add FFT convolution per channel (memory-friendly)."""
    out = np.zeros_like(bus, dtype=np.float64)
    for ch in range(2):
        y = signal.oaconvolve(bus[ch].astype(np.float64), ir[ch])
        out[ch] = y[: bus.shape[1]]
    return out


# ----------------------------------------------------------------------------
# Loudness / peak utilities
# ----------------------------------------------------------------------------
def k_weighting_sos(sr=SR):
    """ITU-R BS.1770 K-weighting as two biquads (pre-filter high shelf + RLB HP)."""
    # High shelf (stage 1)
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    sos1 = np.array(b + a)
    # RLB high-pass (stage 2)
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + K / Q + K * K
    b = [1.0 / a0, -2.0 / a0, 1.0 / a0]
    a = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    sos2 = np.array(b + a)
    return np.vstack([sos1, sos2])


def integrated_lufs(stereo, sr=SR):
    """BS.1770-4 integrated loudness with absolute (−70) and relative (−10) gates."""
    sos = k_weighting_sos(sr)
    blk = int(0.4 * sr)
    hop = int(0.1 * sr)
    n = stereo.shape[1]
    # filter in chunks to limit memory
    powers = []
    z = [signal.sosfilt_zi(sos) * 0.0 for _ in range(2)]
    chunk = sr * 120
    filt = np.zeros((2, n), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        for ch in range(2):
            y, z[ch] = signal.sosfilt(sos, stereo[ch, a:b].astype(np.float64), zi=z[ch])
            filt[ch, a:b] = y
    sq = (filt[0].astype(np.float64) ** 2 + filt[1].astype(np.float64) ** 2)
    cs = np.concatenate(([0.0], np.cumsum(sq)))
    starts = np.arange(0, n - blk + 1, hop)
    ms = (cs[starts + blk] - cs[starts]) / blk
    lk = -0.691 + 10.0 * np.log10(ms + 1e-20)
    gated = lk > -70.0
    if not np.any(gated):
        return -np.inf
    rel = -0.691 + 10.0 * np.log10(np.mean(ms[gated])) - 10.0
    keep = gated & (lk > rel)
    if not np.any(keep):
        return -np.inf
    return -0.691 + 10.0 * np.log10(np.mean(ms[keep]))


def true_peak_dbtp(stereo, sr=SR):
    """4× oversampled peak (BS.1770 true-peak estimate), chunked."""
    peak = 0.0
    chunk = sr * 60
    n = stereo.shape[1]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        seg = stereo[:, max(a - 64, 0):min(b + 64, n)].astype(np.float64)
        up = signal.resample_poly(seg, 4, 1, axis=1)
        peak = max(peak, float(np.max(np.abs(up))))
    return 20.0 * np.log10(peak + 1e-12)


def soft_limiter(stereo, ceiling=0.891, knee=0.08):
    """Transparent tanh soft-knee limiter applied only above (ceiling-knee).
    With the mix designed to peak near −3 dBFS it is effectively inactive."""
    x = stereo.astype(np.float64)
    thr = ceiling - knee
    mag = np.abs(x)
    over = mag > thr
    if np.any(over):
        y = thr + knee * np.tanh((mag[over] - thr) / knee)
        x[over] = np.sign(x[over]) * y
    return x.astype(np.float32)
