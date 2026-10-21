"""
verify.py — independent proof that the rendered audio does what the design says.

Reads the rendered WAV plus the JSON manifest written by session.py, then:
  1. For every analysis point (each stage edge/midpoint and every 60 s) takes a
     20-second window, FFTs left and right separately (Hann, zero-padded to a
     0.011 Hz bin), finds the left and right carrier peaks and reports the
     measured beat = |f_R − f_L| against the designed beat.
  2. For isochronic layers, band-passes around the carrier, takes the Hilbert
     envelope and FFTs it to read the pulse rate directly.
  3. Measures integrated loudness (BS.1770-4), true peak (4× oversampled),
     DC offset, sample clipping, and interaural level balance of the carriers.
  4. Renders a full-track spectrogram and a designed-vs-measured beat plot.
Writes docs/VERIFICATION.md and docs/*.png.

Usage: python3 generator/verify.py output/<name>.wav output/<name>.manifest.json docs/
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth as S  # noqa: E402


def lane_value(points, t):
    return float(S.Lane([tuple(p) for p in points]).at(np.array([t]))[0])


def carrier_peak(x, sr, f_expect, search=3.0, nfft=None):
    n = len(x)
    nfft = nfft or int(2 ** np.ceil(np.log2(n * 4)))
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x * w, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    lo, hi = np.searchsorted(freqs, [f_expect - search, f_expect + search])
    k = lo + int(np.argmax(X[lo:hi]))
    # parabolic interpolation for sub-bin accuracy
    if 0 < k < len(X) - 1:
        a, b, c = np.log(X[k - 1] + 1e-20), np.log(X[k] + 1e-20), np.log(X[k + 1] + 1e-20)
        p = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
    else:
        p = 0.0
    f = freqs[k] + p * (freqs[1] - freqs[0])
    amp = X[k] / (np.sum(w) / 2.0)
    return float(f), float(amp)


def envelope_rate(x, sr, f_center, rate_expect, bw=25.0):
    y = S.bandpass(x, max(f_center - bw, 20.0), f_center + bw, sr, order=4)
    env = np.abs(signal.hilbert(y))
    env = env - np.mean(env)
    nfft = int(2 ** np.ceil(np.log2(len(env) * 4)))
    E = np.abs(np.fft.rfft(env * np.hanning(len(env)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    lo, hi = np.searchsorted(freqs, [max(rate_expect * 0.6, 0.3), rate_expect * 1.4 + 1.0])
    k = lo + int(np.argmax(E[lo:hi]))
    return float(freqs[k])


def main(wav_path, manifest_path, out_dir):
    with open(manifest_path) as f:
        man = json.load(f)
    sr, data = wavfile.read(wav_path)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    x = data.T  # (2, n)
    n = x.shape[1]
    dur = n / sr
    assert sr == man["sr"], "sample-rate mismatch"
    os.makedirs(out_dir, exist_ok=True)

    win = 20.0
    half = int(win * sr / 2)
    times = set()
    for st in man["stages"]:
        times.update([st["start"] + 12.0, 0.5 * (st["start"] + st["end"]), st["end"] - 12.0])
    times.update(np.arange(30.0, dur - 30.0, 60.0).tolist())
    times = sorted(t for t in times if half / sr + 1 < t < dur - half / sr - 1)

    rows = []
    beat_designed, beat_measured, beat_times = [], [], []
    for t in times:
        c = int(t * sr)
        segL = x[0, c - half:c + half].astype(np.float64)
        segR = x[1, c - half:c + half].astype(np.float64)
        for car in man["carriers"]:
            g = lane_value(car["gain_lane"], t)
            if g < 0.02:
                continue
            beat_d = lane_value(car.get("beat_lane") or man["beat_lane"], t)
            fL, aL = carrier_peak(segL, sr, car["hz"] - beat_d / 2)
            fR, aR = carrier_peak(segR, sr, car["hz"] + beat_d / 2)
            meas = abs(fR - fL)
            bal = 20 * np.log10((aL + 1e-12) / (aR + 1e-12))
            rows.append((t, f"binaural {car['hz']:.2f} Hz", beat_d, meas, bal))
            if car is man["carriers"][0]:
                beat_designed.append(beat_d); beat_measured.append(meas); beat_times.append(t)
        for iso in man.get("isochronic", []):
            g = lane_value(iso["gain_lane"], t)
            if g < 0.02:
                continue
            r_d = lane_value(iso["rate_lane"], t)
            r_m = envelope_rate(segL + segR, sr, iso["hz"], r_d)
            rows.append((t, f"isochronic {iso['hz']:.1f} Hz", r_d, r_m, 0.0))

    lufs = S.integrated_lufs(x, sr)
    tp = S.true_peak_dbtp(x, sr)
    dc = [float(np.mean(x[ch])) for ch in range(2)]
    clip = int(np.sum(np.abs(x) >= 0.999))
    peak = float(np.max(np.abs(x)))

    # ---- report
    worst = 0.0
    lines = [f"# Verification report — {os.path.basename(wav_path)}", "",
             f"Duration: {dur/60:.2f} min · Sample rate: {sr} Hz · Channels: 2", "",
             "## Mastering", "",
             f"| Metric | Value |", "|---|---|",
             f"| Integrated loudness (BS.1770-4) | {lufs:.2f} LUFS |",
             f"| True peak (4× oversampled) | {tp:.2f} dBTP |",
             f"| Sample peak | {20*np.log10(peak+1e-12):.2f} dBFS |",
             f"| Samples at/over 0 dBFS | {clip} |",
             f"| DC offset L / R | {dc[0]:.2e} / {dc[1]:.2e} |", "",
             "## Beat / pulse rate: designed vs measured", "",
             "Each row is a 20-second window.  `beat` is |f_right − f_left| of the carrier peaks; `balance` is the interaural level difference of the carrier (should be ~0 dB).", "",
             "| t (min) | layer | designed Hz | measured Hz | error Hz | L−R balance dB |", "|---|---|---|---|---|---|"]
    for (t, name, d, m, bal) in rows:
        err = m - d
        worst = max(worst, abs(err))
        lines.append(f"| {t/60:.2f} | {name} | {d:.3f} | {m:.3f} | {err:+.3f} | {bal:+.2f} |")
    lines += ["", f"Largest beat-rate error across all windows: **{worst:.3f} Hz** (a 20 s window has 0.05 Hz intrinsic resolution; ramping sections show the mean over the window).", ""]

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.5))
    tt = np.linspace(0, dur, 2000)
    ax.plot(tt / 60, S.Lane([tuple(p) for p in man["beat_lane"]]).at(tt), label=f"designed binaural beat, {man['carriers'][0]['hz']:.0f} Hz carrier (Hz)", lw=2)
    ax.plot(np.array(beat_times) / 60, beat_measured, "o", ms=4, label="measured from audio")
    for car in man["carriers"][1:]:
        if car.get("beat_lane"):
            ax.plot(tt / 60, S.Lane([tuple(p) for p in car["beat_lane"]]).at(tt), "--", lw=1.2, label=f"designed beat, {car['hz']:.0f} Hz carrier")
    for iso in man.get("isochronic", []):
        ax.plot(tt / 60, S.Lane([tuple(p) for p in iso["rate_lane"]]).at(tt) * (S.Lane([tuple(p) for p in iso["gain_lane"]]).at(tt) > 0.02), ":", label=f"isochronic {iso['hz']:.0f} Hz pulse rate")
    for st in man["stages"]:
        ax.axvline(st["start"] / 60, color="k", alpha=0.15)
        ax.text(st["start"] / 60 + 0.1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1, st["name"], fontsize=7, rotation=90, va="top")
    ax.set_xlabel("minutes"); ax.set_ylabel("Hz"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.set_title("Entrainment frequency map — designed vs measured")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "beat_map.png"), dpi=110); plt.close()

    # spectrogram (decimated to 8 kHz for the overview)
    dec = 6
    mono = signal.decimate(x[0].astype(np.float64), dec, ftype="fir", zero_phase=True)
    f, t, Sxx = signal.spectrogram(mono, sr // dec, nperseg=4096, noverlap=2048)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.pcolormesh(t / 60, f, 10 * np.log10(Sxx + 1e-16), shading="auto", vmin=-130, vmax=-40, cmap="magma")
    ax.set_ylim(0, 2500); ax.set_xlabel("minutes"); ax.set_ylabel("Hz (left channel)")
    for st in man["stages"]:
        ax.axvline(st["start"] / 60, color="w", alpha=0.3)
        ax.text(st["start"] / 60 + 0.1, 2400, st["name"], color="w", fontsize=7, rotation=90, va="top")
    ax.set_title("Spectrogram (left channel, 0–2.5 kHz)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "spectrogram.png"), dpi=110); plt.close()

    lines += ["## Figures", "", "![beat map](beat_map.png)", "", "![spectrogram](spectrogram.png)", ""]
    with open(os.path.join(out_dir, "VERIFICATION.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"... {len(rows)} measurement rows; worst beat error {worst:.3f} Hz")
    return worst


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
