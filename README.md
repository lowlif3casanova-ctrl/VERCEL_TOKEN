# Pineal Gateway — a procedurally synthesised binaural journey

A 48-minute brainwave-entrainment session (plus a 15-minute daily version) built
from first principles in Python: binaural carriers in the psychoacoustic sweet
spot, isochronic reinforcement, a 4.5 Hz shamanic frame drum with the
Foundation-for-Shamanic-Studies call-back, Tibetan singing bowls and a temple
bell synthesised from their measured mode ratios, a Gyuto-style chord drone,
didgeridoo, dungchen horn swells, distant thunder, formant-synthesised sacred
vowel chant (the Greek seven-vowel ladder and the Bön A-OM-HUM), a tanpura Om
drone, a breath pacer and a pink-noise bed — all tuned to 136.10 Hz (the "Om"
tone, C# in A = 432 tuning) in pure ratios.

Everything is deterministic (fixed seeds) and reproducible from this repository.

## Files

| Path | What it is |
|---|---|
| `generator/synth.py` | Instruments and DSP (oscillators, bowls, drum, chant, drones, reverb, BS.1770 loudness) |
| `generator/session.py` | The stage table / automation / events for both sessions, the mix and mastering, file export |
| `generator/verify.py` | Independent analysis of the rendered audio: measured beat rate per stage vs. design, loudness, true peak, spectrogram |
| `generator/dossier.py` | Builds `docs/RESEARCH_DOSSIER.md` from the research-workflow JSON |
| `docs/LISTENING_GUIDE.md` | How to use the track, what to expect, safety, the stage-by-stage journey script |
| `docs/DESIGN.md` | Every frequency, rate and layer with the tradition or paper it comes from, and an honest evidence note |
| `docs/RESEARCH_DOSSIER.md` | The full research behind the design: six lenses, fact-checked, with sources |
| `docs/VERIFICATION.md` | Measured proof that the audio does what the design says |

## Render it yourself

```bash
pip install numpy scipy matplotlib        # ffmpeg is used for FLAC/MP3 export if present
python3 generator/session.py full  output/pineal_gateway_full     # ~15 min on 4 cores
python3 generator/session.py daily output/pineal_gateway_daily    # ~5 min
python3 generator/verify.py output/pineal_gateway_full.wav output/pineal_gateway_full.manifest.json docs/
```

Outputs: `.wav` (48 kHz float32 master), `.flac` (24-bit lossless), `.mp3`
(320 kbps, forced left/right stereo so the interaural phase that carries the
binaural beat is never folded by joint-stereo encoding), and a `.manifest.json`
the verifier reads.

Rendered audio is not committed (it is large and fully reproducible).

## Honest framing

Stereo headphones are required for the binaural layers. The track is designed
to produce a noticeable shift of state (relaxation, slowed time, hypnagogic
imagery, the "body asleep, mind awake" feeling) on a first listen, and it uses
the real sonic technologies of many traditions. What it cannot honestly promise
is anything beyond that: no sound has been shown to change the pineal gland,
and no recording can guarantee visions, manifestation or travel between realms.
`docs/DESIGN.md` marks every element with its evidence level so you know which
parts are measured physiology and which are a tradition's own teaching.
