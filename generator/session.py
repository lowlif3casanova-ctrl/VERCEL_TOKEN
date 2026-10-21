"""
session.py — the Pineal Gateway session: stage table, automation, events, mix,
mastering, and file output.

    python3 generator/session.py full   output/pineal_gateway_full
    python3 generator/session.py daily  output/pineal_gateway_daily

Writes <out>.wav (float32), <out>.manifest.json (for verify.py), and — if ffmpeg
is present — <out>.flac (24-bit) and <out>.mp3 (320 kbps, forced L/R stereo so
the interaural phase that carries the binaural beat is never folded by joint-
stereo encoding).

All randomness is seeded: the render is bit-for-bit reproducible.

Tonal system
------------
Everything is built on 136.10 Hz: Hans Cousto's "Earth-year" tone (the sidereal
year raised 32 octaves), the pitch most sound-healing lineages use for Om and
for tanpura drones.  It is C# in a tuning where A = 432.1 Hz, so the whole
session is natively in "432 tuning".  All intervals are pure ratios from that
centre (Pythagorean / Timaeus / Chinese lü chain), so nothing beats harshly:

    136.10  1:1   Om / Sa          272.20  2:1   octave
    153.11  9:8   whole tone       306.22  9:4
    170.13  5:4   major third      340.25  5:2
    181.47  4:3   fourth           362.93  8:3
    204.15  3:2   fifth            408.30  3:1   (primary binaural carrier: the
    229.67  27:16 Pythagorean 6th                 350-500 Hz sweet spot of
    255.19  15:8  major seventh                   Oster / Licklider)
                                   544.40  4:1   (40 Hz gamma isochronic carrier)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth as S  # noqa: E402

SR = S.SR
MIN = 60.0

OM = 136.10
R = dict(sa=1.0, re=9 / 8, ga=5 / 4, ma=4 / 3, pa=3 / 2, dha=27 / 16, ni=15 / 8, octave=2.0)
TONE = {k: OM * v for k, v in R.items()}
GYUTO = OM / 2.0          # 68.05 Hz: Gyuto chant fundamentals sit at ~60-75 Hz
BONSHO = 137.8            # measured Hōjōbō temple bell (fits the Om centre)
SOLFEGGIO_963 = 963.0     # Horowitz's "crown/pineal" tone: used once, ceremonially, in the clear
# just-intonation chakra ladder transposed onto the Om octave (Goldman-style C256 set)
CHAKRA_LADDER = [272.2 * r for r in (1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8)]
# Greek seven-vowel / seven-planet ladder (Demetrius, PGM, Discourse on the Eighth and Ninth)
SEVEN_VOWELS = [("A", TONE["sa"]), ("E", TONE["re"]), ("EH", TONE["ga"]), ("I", TONE["ma"]),
                ("O", TONE["pa"]), ("U", TONE["dha"]), ("OM", TONE["ni"])]


def _stage(name, a, b, purpose, frame):
    return dict(name=name, start=a * MIN, end=b * MIN, purpose=purpose, frame=frame)


# ----------------------------------------------------------------------------
# Full 48-minute session
# ----------------------------------------------------------------------------
def design_full():
    D = 48 * MIN
    stages = [
        _stage("Threshold", 0, 4, "Resonant tuning: alpha 10 Hz on the Om carriers; breath pacer 8 -> 6 per minute; tingsha, bonshō and Om bowls panned slightly right (Hatha Yoga Pradipika: listen in the right ear).",
               "Harae / Monroe 'resonant tuning' / Nada yoga 'tinkling -> bell'"),
        _stage("Descent", 4, 11, "Beat glides 10 -> 7.83 Hz (the Schumann fundamental) over five minutes; Gyuto-style chord drone at 68.05 Hz with the 5th and 10th harmonics lifted; dungchen swell. Body goes to sleep, mind stays.",
               "Monroe Focus 10 'mind awake, body asleep' / Gyuto dbyangs"),
        _stage("Drum Gate", 11, 21, "Distant thunder opens the gate; frame drum at 4.25 Hz (255 bpm, Harner's measured journey tempo and the Earth-year theta octave) with a theta isochronic pulse at the same rate; didgeridoo breath-drone; stone-chamber drone; beat settles at 4.25 Hz.",
               "Core shamanism (Harner / Siberian / Sámi) / Yolngu yidaki / Neolithic chamber"),
        _stage("Ajna", 21, 29, "Seven Greek vowels A E Ē I O U Ō, each a pure-ratio step higher, then IAŌ; the just-intonation chakra bowl ladder rises 272 -> 510 Hz with each bowl 'wah'-ing at the 6 Hz theta beat.",
               "Greek Magical Papyri / Discourse on the Eighth and Ninth / Shat-chakra-nirupana"),
        _stage("No-Time", 29, 37, "Outer carriers at 3.0 -> 2.13 Hz with the 136 Hz carrier at 10 Hz alpha (Monroe-style stacking: body deeply asleep, awareness awake) and four 90-second 40 Hz gamma cues (Lutz 2004 monk gamma, ASSR optimum); dungchen pedal tones; a 0.2 Hz surge on the drone; the 'thousand thunders' pass by at the end.",
               "Monroe Focus 15 'no time' / Cade-Wise awakened mind / Chönyid bardo"),
        _stage("Crown", 37, 43, "Clear-light return: the beat rises to 7.83 Hz, A-OM-HUM three-point chant (272 / 204 / 136 Hz: forehead, throat, heart), Gyuto drone, conch clusters, a whisper of 3-8 kHz 'air'.",
               "Bön / Dzogchen A-OM-HUM / Hesychast light / Tomatis"),
        _stage("Return", 43, 48, "Call-back drum signal, 963 Hz bowl struck once in the clear, beat climbs 7.83 -> 10 -> 14.1 Hz (SMR) so you come back alert; breath pacer returns; Om bowls and tingsha close the circle.",
               "Foundation for Shamanic Studies call-back / Sterman SMR / Yoga Nidra externalisation"),
    ]
    # Beat-rate ladder.  4.25 Hz = Harner's measured journey tempo (254.7 bpm) and Cousto's
    # Earth-year theta octave (136.10 / 32); 2.13 Hz = the Earth-year delta octave (136.10 / 64),
    # the Focus-15 'no time' floor; 7.83 Hz = Schumann fundamental; 14.1 Hz = Schumann 2nd mode / SMR.
    beat = [(0, 10.0), (4 * MIN, 10.0), (9 * MIN, 7.83), (11 * MIN, 7.83), (13.5 * MIN, 4.25),
            (21 * MIN, 4.25), (23 * MIN, 6.0), (29 * MIN, 6.0), (31 * MIN, 3.0), (33 * MIN, 3.0), (34 * MIN, 2.13),
            (35.5 * MIN, 2.13), (37 * MIN, 4.25), (40 * MIN, 7.83), (43.5 * MIN, 7.83), (45.3 * MIN, 10.0),
            (47.6 * MIN, 14.1), (D, 14.1)]
    # The Om-fundamental carrier keeps a 10 Hz alpha beat during No-Time (Monroe-style stacking:
    # body deeply asleep on the outer carriers, awareness held awake on the inner one) and keeps a
    # 7.83 Hz theta bridge under the alpha/SMR return (Cade-Wise 'awakened mind' = all bands at once).
    beat_om = [p for p in beat if p[0] <= 29 * MIN] + [(31 * MIN, 10.0), (37 * MIN, 10.0), (40 * MIN, 7.83),
                                                       (46.0 * MIN, 7.83), (47.7 * MIN, 12.5), (D, 12.5)]
    carriers = [
        dict(hz=3 * OM, gain=[(0, 0), (25, 0.17), (D - 45, 0.17), (D - 6, 0.0)], beat=None),
        dict(hz=2 * OM, gain=[(0, 0), (25, 0.12), (D - 45, 0.12), (D - 6, 0.0)], beat=None),
        dict(hz=OM,     gain=[(0, 0), (25, 0.10), (D - 45, 0.10), (D - 6, 0.0)], beat=beat_om),
    ]
    isochronic = [
        dict(name="iso_theta", hz=TONE["pa"], rate="beat",
             gain=[(0, 0), (12.5 * MIN, 0), (14 * MIN, 0.05), (20.5 * MIN, 0.05), (21.5 * MIN, 0),
                   (29.5 * MIN, 0), (31 * MIN, 0.045), (36 * MIN, 0.045), (37 * MIN, 0)]),
        # 40 Hz gamma as 90-second 'lucidity cues' (Voss 2014 / Chan 2022 pulse logic), not a continuous bed
        dict(name="iso_gamma", hz=4 * OM, rate=[(0, 40.0)],
             gain=[(0, 0)] + sum([[(s, 0), (s + 20, 0.055), (s + 70, 0.055), (s + 90, 0)]
                                  for s in (31.8 * MIN, 34.3 * MIN, 36.6 * MIN, 41.0 * MIN)], [])),
    ]
    drums = [dict(start=11.5 * MIN, end=20.5 * MIN, rate_hz=4.25, gain=0.42, accent_every=4, humanize_ms=7, ramp_s=25)]
    callback_at = 43 * MIN + 5
    bowls = []
    # Threshold: Om bowls, right-leaning
    bowls += [dict(t=60, f0=OM, gain=0.42, pan=0.25, split=8.0), dict(t=180, f0=OM, gain=0.42, pan=0.2, split=8.0)]
    # Descent: fifth and octave, wah at Schumann rate
    bowls += [dict(t=6 * MIN, f0=TONE["pa"], gain=0.36, pan=-0.3, split=7.83), dict(t=8.5 * MIN, f0=TONE["octave"], gain=0.36, pan=0.3, split=7.83)]
    # Drum Gate
    bowls += [dict(t=15 * MIN, f0=OM, gain=0.36, pan=-0.2, split=4.25), dict(t=18 * MIN, f0=TONE["pa"], gain=0.34, pan=0.25, split=4.25)]
    # Ajna: chakra ladder in just intonation, each wah-ing at 6 Hz
    for i, f in enumerate(CHAKRA_LADDER):
        bowls.append(dict(t=21.6 * MIN + i * 55, f0=f, gain=0.34 - 0.012 * i, pan=(-1) ** i * 0.35, split=6.0))
    # No-Time
    bowls += [dict(t=32 * MIN, f0=OM, gain=0.34, pan=-0.2, split=3.0), dict(t=35 * MIN, f0=TONE["octave"], gain=0.32, pan=0.2, split=2.13)]
    # Crown
    bowls += [dict(t=38 * MIN, f0=TONE["octave"], gain=0.36, pan=0.25, split=7.83), dict(t=40 * MIN, f0=3 * OM, gain=0.3, pan=-0.25, split=7.83),
              dict(t=41.5 * MIN, f0=4 * OM, gain=0.26, pan=0.2, split=7.83), dict(t=42.5 * MIN, f0=OM, gain=0.4, pan=0.0, split=7.83)]
    # Return: the single 963 Hz strike in the clear, then Om bowls home
    bowls += [dict(t=45 * MIN + 20, f0=SOLFEGGIO_963, gain=0.3, pan=0.15, split=None),
              dict(t=46 * MIN, f0=OM, gain=0.42, pan=0.2, split=10.0), dict(t=46.8 * MIN, f0=OM, gain=0.42, pan=0.2, split=10.0),
              dict(t=47.5 * MIN, f0=OM, gain=0.44, pan=0.2, split=None)]
    bonsho = [dict(t=12, gain=0.36), dict(t=4 * MIN, gain=0.34), dict(t=11 * MIN, gain=0.34), dict(t=21 * MIN, gain=0.34),
              dict(t=29 * MIN, gain=0.32), dict(t=37 * MIN, gain=0.36), dict(t=43 * MIN, gain=0.34)]
    tingsha = [dict(t=8, gain=0.22), dict(t=2 * MIN, gain=0.18), dict(t=21 * MIN + 3, gain=0.16), dict(t=37 * MIN + 3, gain=0.18),
               dict(t=47 * MIN + 50, gain=0.22), dict(t=47 * MIN + 53, gain=0.2)]
    dungchen = [dict(t=9.3 * MIN, gain=0.22), dict(t=30 * MIN, gain=0.22), dict(t=33.5 * MIN, gain=0.22), dict(t=36 * MIN, gain=0.2)]
    thunder = [dict(t=10.9 * MIN, gain=0.26), dict(t=35.4 * MIN, gain=0.2), dict(t=36.0 * MIN, gain=0.24), dict(t=36.5 * MIN, gain=0.22)]
    # Chavín-style conch cluster: the upper-world call during the Crown ascent (beats at 3.9 / 7.83 Hz)
    pututu = [dict(t=38.6 * MIN, gain=0.2, pan=-0.4), dict(t=40.4 * MIN, gain=0.2, pan=0.4), dict(t=42.0 * MIN, gain=0.18, pan=0.0)]
    drones = dict(
        tanpura=dict(f0=OM, gain=[(0, 0), (40, 0.16), (42.5 * MIN, 0.16), (45 * MIN, 0.0)]),
        # Neolithic-chamber register (108.88 Hz = 5:4 below Om), heavily reverberant, under the drum journey
        chamber=dict(f0=OM * 4 / 5, gain=[(0, 0), (11.5 * MIN, 0), (13 * MIN, 0.07), (20 * MIN, 0.07), (21 * MIN, 0)]),
        gyuto=dict(f0=GYUTO, gain=[(0, 0), (4.8 * MIN, 0), (6 * MIN, 0.10), (10.5 * MIN, 0.10), (11.5 * MIN, 0),
                                   (37.5 * MIN, 0), (38.5 * MIN, 0.09), (42 * MIN, 0.09), (43 * MIN, 0)],
                   # Smith/Stevens/Tomlinson 1967: the Gyuto 'chord' lifts the 5th and 10th harmonics
                   formant=[(0, 5.0), (6 * MIN, 5.0), (7.5 * MIN, 10.0), (9 * MIN, 5.0), (10.5 * MIN, 10.0), (11.5 * MIN, 5.0),
                            (38.5 * MIN, 5.0), (40 * MIN, 10.0), (41.5 * MIN, 5.0), (43 * MIN, 10.0)]),
        didgeridoo=dict(f0=GYUTO, gain=[(0, 0), (12 * MIN, 0), (13.5 * MIN, 0.10), (20 * MIN, 0.10), (21 * MIN, 0)], breath_hz=0.55),
    )
    seven = []
    t0 = 0.0
    for (vw, f) in SEVEN_VOWELS:
        seven.append((t0, t0 + 50.0, vw, f)); t0 += 50.0
    seven += [(t0 + 2, t0 + 12, "I", TONE["octave"]), (t0 + 12, t0 + 24, "A", TONE["octave"]), (t0 + 24, t0 + 40, "OM", TONE["octave"])]
    aum = []
    t0 = 0.0
    for _ in range(15):   # A (forehead, 272) - OM (throat, 204) - HUM (heart, 136): 18 s per round
        aum += [(t0, t0 + 6.0, "A", TONE["octave"]), (t0 + 6.0, t0 + 12.0, "OM", TONE["pa"]), (t0 + 12.0, t0 + 15.0, "U", OM), (t0 + 15.0, t0 + 18.0, "M", OM)]
        t0 += 18.0
    chant = [
        dict(start=21.4 * MIN, gain=[(0, 0), (21.4 * MIN, 0), (21.9 * MIN, 0.13), (28.2 * MIN, 0.13), (28.6 * MIN, 0)], seq=seven, fade_s=1.5),
        dict(start=37.6 * MIN, gain=[(0, 0), (37.6 * MIN, 0), (38.1 * MIN, 0.12), (41.9 * MIN, 0.12), (42.3 * MIN, 0)], seq=aum, fade_s=1.0),
    ]
    breath = dict(rate=[(0, 8 / 60), (3 * MIN, 6 / 60), (44 * MIN, 6 / 60), (47.5 * MIN, 7 / 60)],
                  gain=[(0, 0), (30, 0.06), (11 * MIN, 0.06), (13 * MIN, 0.0), (43.5 * MIN, 0), (44.5 * MIN, 0.05), (D - 20, 0.05), (D - 2, 0)])
    pink = dict(gain=[(0, 0), (15, 0.011), (D - 30, 0.011), (D - 3, 0)])
    air = dict(gain=[(0, 0), (38 * MIN, 0), (39 * MIN, 0.004), (42 * MIN, 0.004), (42.8 * MIN, 0)])
    # Monroe's 'vibrational state': a slow 0.2 Hz (5-second) surge on the drone bed, deepening through No-Time
    vibration = dict(rate_hz=0.2, depth=[(0, 0), (30.5 * MIN, 0), (33 * MIN, 0.35), (36.5 * MIN, 0.35), (37.5 * MIN, 0)])
    return dict(name="Pineal Gateway — full session (48 min)", duration=D, stages=stages, beat=beat, carriers=carriers,
                isochronic=isochronic, drums=drums, callback_at=callback_at, bowls=bowls, bonsho=bonsho, tingsha=tingsha,
                dungchen=dungchen, thunder=thunder, pututu=pututu, drones=drones, chant=chant, breath=breath, pink=pink, air=air,
                vibration=vibration)


# ----------------------------------------------------------------------------
# Daily 15-minute session
# ----------------------------------------------------------------------------
def design_daily():
    D = 15 * MIN
    stages = [
        _stage("Threshold", 0, 2, "Alpha 10 Hz, Om carriers, breath pacer", "resonant tuning"),
        _stage("Descent", 2, 5, "Glide to 6 Hz theta; Gyuto drone", "Focus 10"),
        _stage("Drum Gate", 5, 10, "4.25 Hz drum and theta pulse; didgeridoo; chamber drone", "journey"),
        _stage("No-Time", 10, 12.5, "3 Hz with 10 Hz alpha on the Om carrier and 40 Hz gamma; A-OM-HUM", "Focus 15 / awakened mind"),
        _stage("Return", 12.5, 15, "Call-back; 7.83 -> 14.1 Hz; 963 Hz bowl in the clear", "call-back / SMR"),
    ]
    beat = [(0, 10.0), (2 * MIN, 10.0), (4.5 * MIN, 6.0), (5 * MIN, 6.0), (6 * MIN, 4.25), (10 * MIN, 4.25),
            (10.8 * MIN, 3.0), (12.3 * MIN, 3.0), (13.2 * MIN, 7.83), (14.6 * MIN, 14.1), (D, 14.1)]
    beat_om = [p for p in beat if not (10.8 * MIN <= p[0] <= 12.3 * MIN)] + [(10.8 * MIN, 10.0), (12.3 * MIN, 10.0)]
    beat_om.sort()
    carriers = [
        dict(hz=3 * OM, gain=[(0, 0), (15, 0.17), (D - 30, 0.17), (D - 5, 0.0)], beat=None),
        dict(hz=2 * OM, gain=[(0, 0), (15, 0.12), (D - 30, 0.12), (D - 5, 0.0)], beat=None),
        dict(hz=OM,     gain=[(0, 0), (15, 0.10), (D - 30, 0.10), (D - 5, 0.0)], beat=beat_om),
    ]
    isochronic = [
        dict(name="iso_theta", hz=TONE["pa"], rate="beat",
             gain=[(0, 0), (5.5 * MIN, 0), (6.5 * MIN, 0.05), (12 * MIN, 0.05), (12.6 * MIN, 0)]),
        dict(name="iso_gamma", hz=4 * OM, rate=[(0, 40.0)],
             gain=[(0, 0), (10 * MIN, 0), (11 * MIN, 0.055), (12.3 * MIN, 0.055), (12.9 * MIN, 0)]),
    ]
    drums = [dict(start=5.3 * MIN, end=9.8 * MIN, rate_hz=4.25, gain=0.42, accent_every=4, humanize_ms=7, ramp_s=15)]
    callback_at = 12.55 * MIN          # ends ~13:55 with the shorter 30 s grounding used in the daily version
    bowls = [dict(t=20, f0=OM, gain=0.42, pan=0.25, split=8.0), dict(t=80, f0=OM, gain=0.4, pan=0.2, split=8.0),
             dict(t=3.5 * MIN, f0=TONE["pa"], gain=0.36, pan=-0.3, split=6.0), dict(t=7.5 * MIN, f0=OM, gain=0.36, pan=-0.2, split=4.25)]
    for i, f in enumerate((CHAKRA_LADDER[0], CHAKRA_LADDER[4], CHAKRA_LADDER[6])):
        bowls.append(dict(t=10.2 * MIN + i * 40, f0=f, gain=0.32, pan=(-1) ** i * 0.3, split=3.0))
    # 963 Hz bowl once the tanpura has faded (13:36) and the grounding strokes have ended (~13:55)
    bowls += [dict(t=14.0 * MIN, f0=SOLFEGGIO_963, gain=0.28, pan=0.15, split=None),
              dict(t=14.35 * MIN, f0=OM, gain=0.42, pan=0.2, split=10.0), dict(t=14.6 * MIN, f0=OM, gain=0.44, pan=0.2, split=None)]
    bonsho = [dict(t=10, gain=0.34), dict(t=2 * MIN, gain=0.32), dict(t=5 * MIN, gain=0.32), dict(t=10 * MIN, gain=0.32), dict(t=12.5 * MIN, gain=0.32)]
    tingsha = [dict(t=6, gain=0.22), dict(t=10 * MIN + 3, gain=0.16), dict(t=14 * MIN + 50, gain=0.22), dict(t=14 * MIN + 53, gain=0.2)]
    dungchen = [dict(t=4.2 * MIN, gain=0.22), dict(t=10.3 * MIN, gain=0.22)]
    thunder = [dict(t=4.9 * MIN, gain=0.24), dict(t=12.0 * MIN, gain=0.2)]
    pututu = [dict(t=11.4 * MIN, gain=0.2, pan=-0.4), dict(t=12.1 * MIN, gain=0.18, pan=0.4)]
    drones = dict(
        tanpura=dict(f0=OM, gain=[(0, 0), (30, 0.16), (12.9 * MIN, 0.16), (13.6 * MIN, 0.0)]),
        chamber=dict(f0=OM * 4 / 5, gain=[(0, 0), (5.5 * MIN, 0), (6.5 * MIN, 0.07), (9.5 * MIN, 0.07), (10.2 * MIN, 0)]),
        gyuto=dict(f0=GYUTO, gain=[(0, 0), (2.2 * MIN, 0), (3 * MIN, 0.10), (4.8 * MIN, 0.10), (5.5 * MIN, 0)],
                   formant=[(0, 5.0), (3 * MIN, 5.0), (4 * MIN, 10.0), (5 * MIN, 5.0)]),
        didgeridoo=dict(f0=GYUTO, gain=[(0, 0), (5.5 * MIN, 0), (6.5 * MIN, 0.10), (9.5 * MIN, 0.10), (10.2 * MIN, 0)], breath_hz=0.55),
    )
    aum = []
    t0 = 0.0
    for _ in range(6):
        aum += [(t0, t0 + 6.0, "A", TONE["octave"]), (t0 + 6.0, t0 + 12.0, "OM", TONE["pa"]), (t0 + 12.0, t0 + 15.0, "U", OM), (t0 + 15.0, t0 + 18.0, "M", OM)]
        t0 += 18.0
    chant = [dict(start=10.4 * MIN, gain=[(0, 0), (10.4 * MIN, 0), (10.9 * MIN, 0.12), (12.1 * MIN, 0.12), (12.5 * MIN, 0)], seq=aum, fade_s=1.0)]
    breath = dict(rate=[(0, 8 / 60), (2 * MIN, 6 / 60), (13 * MIN, 6 / 60), (14.8 * MIN, 7 / 60)],
                  gain=[(0, 0), (20, 0.06), (4.5 * MIN, 0.06), (5.5 * MIN, 0.0), (12.8 * MIN, 0), (13.5 * MIN, 0.05), (D - 15, 0.05), (D - 2, 0)])
    pink = dict(gain=[(0, 0), (10, 0.011), (D - 20, 0.011), (D - 3, 0)])
    air = dict(gain=[(0, 0), (10.5 * MIN, 0), (11 * MIN, 0.004), (12.2 * MIN, 0.004), (12.6 * MIN, 0)])
    vibration = dict(rate_hz=0.2, depth=[(0, 0), (10.5 * MIN, 0), (11.2 * MIN, 0.35), (12.2 * MIN, 0.35), (12.6 * MIN, 0)])
    return dict(name="Pineal Gateway — daily session (15 min)", duration=D, stages=stages, beat=beat, carriers=carriers,
                isochronic=isochronic, drums=drums, callback_at=callback_at, callback_ground_s=30.0, bowls=bowls, bonsho=bonsho,
                tingsha=tingsha, dungchen=dungchen, thunder=thunder, pututu=pututu, drones=drones, chant=chant, breath=breath,
                pink=pink, air=air, vibration=vibration)


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def render(design, out_base, target_lufs=-16.0, ceiling_dbtp=-1.0, seed=2026):
    t_start = time.time()
    sr = SR
    D = design["duration"]
    n = int(D * sr)
    rng = np.random.default_rng(seed)
    log = lambda *a: print(f"[{time.time() - t_start:7.1f}s]", *a, flush=True)

    dry = np.zeros((2, n), dtype=np.float32)   # binaural carriers + iso + noise: never reverberated
    wet = np.zeros((2, n), dtype=np.float32)   # drums, bowls, chant, part of drones: go to the reverb

    beat_lane = S.Lane(design["beat"])
    blk = 10 * sr

    # --- continuous layers, block-wise -------------------------------------
    car_acc = [S.PhaseAcc(2) for _ in design["carriers"]]
    car_gain = [S.Lane(c["gain"]) for c in design["carriers"]]
    car_beat = [S.Lane(c["beat"]) if c.get("beat") else beat_lane for c in design["carriers"]]
    iso_acc = [(S.PhaseAcc(1), S.PhaseAcc(1)) for _ in design["isochronic"]]
    iso_gain = [S.Lane(i["gain"]) for i in design["isochronic"]]
    iso_rate = [None if i["rate"] == "beat" else S.Lane(i["rate"]) for i in design["isochronic"]]

    dr = design["drones"]
    # tanpura: Sa string (136.1) plus the Pa string (204.15), the 5-8-8-1 tuning's fifth
    tanpura = S.AdditiveDrone(dr["tanpura"]["f0"], n_harm=28, rolloff=1.0)
    tanpura_pa = S.AdditiveDrone(dr["tanpura"]["f0"] * 1.5, n_harm=20, rolloff=1.05)
    tan_gain = S.Lane(dr["tanpura"]["gain"])
    tan_lfo_rates = rng.uniform(0.03, 0.25, size=28)
    tan_lfo_phase = rng.uniform(0, S.TWO_PI, size=28)
    pa_lfo_rates = rng.uniform(0.03, 0.2, size=20)
    pa_lfo_phase = rng.uniform(0, S.TWO_PI, size=20)
    gyuto = S.AdditiveDrone(dr["gyuto"]["f0"], n_harm=36, rolloff=0.8)
    gy_gain = S.Lane(dr["gyuto"]["gain"])
    gy_formant = S.Lane(dr["gyuto"]["formant"])
    didg = S.AdditiveDrone(dr["didgeridoo"]["f0"], n_harm=36, rolloff=0.75)
    dg_gain = S.Lane(dr["didgeridoo"]["gain"])
    dg_breath = dr["didgeridoo"]["breath_hz"]
    chamber_acc = S.PhaseAcc(9)
    ch_gain = S.Lane(dr["chamber"]["gain"])
    ch_f0 = dr["chamber"]["f0"]
    breath_gain = S.Lane(design["breath"]["gain"])
    breath_rate = S.Lane(design["breath"]["rate"])
    breath_acc = S.PhaseAcc(1)
    breath_noise = [S.BandNoiseStream(np.random.default_rng(seed + 800 + ch), 110.0, 520.0, sr, order=4) for ch in range(2)]
    pink_gain = S.Lane(design["pink"]["gain"])
    pink_stream = [S.PinkStream(np.random.default_rng(seed + 810 + ch)) for ch in range(2)]
    air_gain = S.Lane(design["air"]["gain"])
    air_stream = S.AirStream(np.random.default_rng(seed + 820), sr)
    vib_depth = S.Lane(design.get("vibration", {}).get("depth", [(0, 0)]))
    vib_rate = design.get("vibration", {}).get("rate_hz", 0.2)

    log("rendering continuous layers")
    for a in range(0, n, blk):
        b = min(a + blk, n)
        t = np.arange(a, b) / sr
        beat = beat_lane.at(t)
        # binaural carriers: left = f - beat/2, right = f + beat/2; exactly equal level, dry, centred
        for k, c in enumerate(design["carriers"]):
            g = car_gain[k].at(t)
            bk = car_beat[k].at(t)
            ph = car_acc[k].advance(np.stack([c["hz"] - bk / 2, c["hz"] + bk / 2]), sr)
            dry[0, a:b] += (g * np.sin(ph[0])).astype(np.float32)
            dry[1, a:b] += (g * np.sin(ph[1])).astype(np.float32)
        # isochronic layers: identical in both ears (monaural), raised-cosine gate
        for k, iso in enumerate(design["isochronic"]):
            g = iso_gain[k].at(t)
            if np.max(g) <= 0:
                continue
            rate = beat if iso_rate[k] is None else iso_rate[k].at(t)
            ph_c = iso_acc[k][0].advance(np.full(b - a, iso["hz"]), sr)
            ph_g = iso_acc[k][1].advance(rate, sr)
            gate = (0.5 - 0.5 * np.cos(ph_g)) ** (3.0 if iso["name"] == "iso_gamma" else 2.0)
            sig = (g * gate * np.sin(ph_c)).astype(np.float32)
            dry[0, a:b] += sig
            dry[1, a:b] += sig
        # tanpura / Om drone with slowly blooming harmonics (jivari); the 0.2 Hz 'vibration' surge rides on it
        g = tan_gain.at(t)
        if np.max(g) > 0:
            lfo = 0.55 + 0.45 * np.sin(S.TWO_PI * tan_lfo_rates[:, None] * t[None, :] + tan_lfo_phase[:, None])
            sig = tanpura.render(t, extra_weights=lfo, vibrato_hz=0.0)
            lfo_pa = 0.55 + 0.45 * np.sin(S.TWO_PI * pa_lfo_rates[:, None] * t[None, :] + pa_lfo_phase[:, None])
            sig = sig + 0.5 * tanpura_pa.render(t, extra_weights=lfo_pa, vibrato_hz=0.0)
            vib = vib_depth.at(t)
            sig = g * sig / 4.6 * (1.0 - vib * 0.5 * (1.0 - np.cos(S.TWO_PI * vib_rate * t)))
            for ch in range(2):
                wet[ch, a:b] += (0.5 * sig).astype(np.float32)
                dry[ch, a:b] += (0.5 * sig).astype(np.float32)
        # Gyuto-style chord drone: formant moves between the 5th and 10th harmonics
        g = gy_gain.at(t)
        if np.max(g) > 0:
            sig = gyuto.render(t, formant_center=gy_formant.at(t), formant_width=1.2, formant_gain=6.0,
                               vibrato_hz=0.21, vibrato_depth=0.0015)
            sig = g * sig / 6.0
            for ch in range(2):
                wet[ch, a:b] += (0.45 * sig).astype(np.float32)
                dry[ch, a:b] += (0.55 * sig).astype(np.float32)
        # didgeridoo: same engine, circular-breathing pulse + wandering formant
        g = dg_gain.at(t)
        if np.max(g) > 0:
            pulse = 0.72 + 0.28 * (0.5 - 0.5 * np.cos(S.TWO_PI * dg_breath * t)) ** 1.5
            wander = 7.0 + 2.5 * np.sin(S.TWO_PI * 0.037 * t) + 1.2 * np.sin(S.TWO_PI * 0.11 * t + 1.0)
            sig = didg.render(t, formant_center=wander, formant_width=1.1, formant_gain=5.0,
                              vibrato_hz=dg_breath * 2, vibrato_depth=0.0012)
            sig = g * pulse * sig / 6.0
            for ch in range(2):
                wet[ch, a:b] += (0.5 * sig).astype(np.float32)
                dry[ch, a:b] += (0.5 * sig).astype(np.float32)
        # stone-chamber drone: mostly reverberant
        g = ch_gain.at(t)
        if np.max(g) > 0:
            sig = g * S.chamber_drone_block(chamber_acc, t, ch_f0, sr) / 2.5
            for ch in range(2):
                wet[ch, a:b] += (0.8 * sig).astype(np.float32)
                dry[ch, a:b] += (0.2 * sig).astype(np.float32)
        else:
            chamber_acc.advance(ch_f0 * np.arange(1, 10)[:, None] * np.ones((1, b - a)), sr)
        # breath pacer: ocean-like filtered noise swelling at the breath-rate lane (phase-continuous)
        g = breath_gain.at(t)
        ph_b = breath_acc.advance(breath_rate.at(t), sr)
        if np.max(g) > 0:
            swell = (0.5 - 0.5 * np.cos(ph_b)) ** 1.6
            for ch in range(2):
                dry[ch, a:b] += (g * swell * breath_noise[ch].next(b - a) * 0.7).astype(np.float32)
        # pink bed (independent per channel, filter state carried across blocks)
        g = pink_gain.at(t)
        if np.max(g) > 0:
            for ch in range(2):
                dry[ch, a:b] += (g * pink_stream[ch].next(b - a)).astype(np.float32)
        # 3-8 kHz 'air' (absolute time axis, carried filter state)
        g = air_gain.at(t)
        if np.max(g) > 0:
            dry[:, a:b] += (g[None, :] * air_stream.next(t)).astype(np.float32)
        if (a // blk) % 30 == 0:
            log(f"  block {a // blk + 1}/{(n + blk - 1) // blk}")

    # --- events ---------------------------------------------------------------
    log("rendering drums")
    wet += S.render_drum_track(n, design["drums"], sr, np.random.default_rng(seed + 1))
    if design.get("callback_at") is not None:
        cb, cb_end = S.harner_callback(n, design["callback_at"], sr, np.random.default_rng(seed + 2), gain=0.46,
                                       ground_dur=design.get("callback_ground_s", 60.0))
        wet += cb
        log(f"  call-back signal {design['callback_at']/60:.2f} -> {cb_end/60:.2f} min")

    log("rendering bowls, bells, horns, thunder")
    for i, bw in enumerate(design["bowls"]):
        sig = S.singing_bowl(bw["f0"], sr, dur=38.0, gain=bw["gain"], rng=np.random.default_rng(seed + 10 + i),
                             rim_sing=0.25, split_hz=bw.get("split"))
        S.add_at(wet, int(bw["t"] * sr), S.pan(sig, bw["pan"]).astype(np.float32))
    for i, ev in enumerate(design["bonsho"]):
        sig = S.singing_bowl(BONSHO, sr, dur=42.0, gain=ev["gain"], rng=np.random.default_rng(seed + 300 + i), rim_sing=0.0,
                             ratios=S.BONSHO_RATIOS, amps=S.BONSHO_AMPS, t60=S.BONSHO_T60)
        S.add_at(wet, int(ev["t"] * sr), S.pan(sig, 0.2).astype(np.float32))
    for i, ev in enumerate(design["tingsha"]):
        sig = S.tingsha(sr, gain=ev["gain"], rng=np.random.default_rng(seed + 400 + i))
        S.add_at(wet, int(ev["t"] * sr), S.pan(sig, 0.3).astype(np.float32))
    for i, ev in enumerate(design["dungchen"]):
        sig = S.dungchen_swell(sr, gain=ev["gain"], rng=np.random.default_rng(seed + 500 + i))
        S.add_at(wet, int(ev["t"] * sr), S.pan(sig, -0.1).astype(np.float32))
    for i, ev in enumerate(design["thunder"]):
        sig = S.thunder_swell(sr, gain=ev["gain"], rng=np.random.default_rng(seed + 600 + i))
        S.add_at(wet, int(ev["t"] * sr), S.pan(sig, (-1) ** i * 0.5).astype(np.float32))
    for i, ev in enumerate(design.get("pututu", [])):
        sig = S.pututu_cluster(sr, gain=ev["gain"], rng=np.random.default_rng(seed + 700 + i))
        S.add_at(wet, int(ev["t"] * sr), S.pan(sig, ev["pan"]).astype(np.float32))

    log("rendering chant")
    for i, ch in enumerate(design["chant"]):
        length = max(e for (_, e, *_rest) in ch["seq"]) + 4.0
        m = int(length * sr)
        voice = S.chant_voice(m, OM, ch["seq"], sr, np.random.default_rng(seed + 50 + i), fade_s=ch.get("fade_s", 1.5))
        t = ch["start"] + np.arange(m) / sr
        g = S.Lane(ch["gain"]).at(t)
        S.add_at(wet, int(ch["start"] * sr), (voice * g[None, :]).astype(np.float32))

    # --- reverb on the wet bus only ------------------------------------------------
    log("reverb")
    from scipy.signal import oaconvolve
    ir = S.synthetic_ir(sr, t60=3.6, rng=np.random.default_rng(seed + 99))
    mix = dry
    for ch in range(2):
        y = oaconvolve(wet[ch].astype(np.float64), ir[ch])[:n]
        mix[ch] += (0.34 * y).astype(np.float32) + wet[ch]
        del y
    del wet

    # --- mastering ---------------------------------------------------------------
    log("mastering")
    # master fade so that no ringing bowl, reverb tail or bell is cut at the last sample
    nf = int(4.0 * sr)
    mix[:, -nf:] *= S.raised_cosine_fade(0, nf, nf)[None, :].astype(np.float32)
    nf0 = int(0.5 * sr)
    mix[:, :nf0] *= S.raised_cosine_fade(nf0, 0, nf0)[None, :].astype(np.float32)
    sos = S.signal.butter(2, 20.0, "hp", fs=sr, output="sos")
    for ch in range(2):
        mix[ch] = S.signal.sosfiltfilt(sos, mix[ch].astype(np.float64)).astype(np.float32)
    lufs = S.integrated_lufs(mix, sr)
    gain_db = target_lufs - lufs
    mix *= np.float32(10 ** (gain_db / 20.0))
    tp = S.true_peak_dbtp(mix, sr)
    if tp > ceiling_dbtp:
        mix *= np.float32(10 ** ((ceiling_dbtp - tp) / 20.0))
        mix = S.soft_limiter(mix, ceiling=10 ** (ceiling_dbtp / 20.0))
    lufs_final = S.integrated_lufs(mix, sr)
    tp_final = S.true_peak_dbtp(mix, sr)
    log(f"  loudness {lufs:.2f} -> {lufs_final:.2f} LUFS (gain {gain_db:+.2f} dB), true peak {tp_final:.2f} dBTP")

    # --- write ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    wav = out_base + ".wav"
    wavfile.write(wav, sr, np.ascontiguousarray(mix.T))
    manifest = dict(
        name=design["name"], sr=sr, duration_s=D, target_lufs=target_lufs, measured_lufs=lufs_final, true_peak_dbtp=tp_final,
        stages=design["stages"], beat_lane=design["beat"],
        carriers=[dict(hz=c["hz"], gain_lane=c["gain"], beat_lane=c.get("beat")) for c in design["carriers"]],
        isochronic=[dict(name=i["name"], hz=i["hz"], gain_lane=i["gain"],
                         rate_lane=(design["beat"] if i["rate"] == "beat" else i["rate"])) for i in design["isochronic"]],
        drums=design["drums"], callback_at=design.get("callback_at"), bowls=design["bowls"], bonsho=design["bonsho"],
        tingsha=design["tingsha"], dungchen=design["dungchen"], thunder=design["thunder"], pututu=design.get("pututu", []),
        drones={k: {kk: vv for kk, vv in v.items()} for k, v in design["drones"].items()},
        chant=[dict(start=c["start"], seq=c["seq"]) for c in design["chant"]],
        breath=design["breath"], seed=seed,
    )
    with open(out_base + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"wrote {wav}")
    encode(wav, out_base, log)
    log("done")
    return manifest


def encode(wav, out_base, log=print):
    title = os.path.basename(out_base).replace("_", " ")
    try:
        comment = ("Binaural: stereo headphones required. iPhone/AirPods: turn OFF Spatialize Stereo, Personalized Spatial Audio, "
                   "Mono Audio and Conversation Awareness; keep L/R balance centred.")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav,
                        "-c:a", "flac", "-sample_fmt", "s32", "-compression_level", "8",
                        "-metadata", "title=" + title, "-metadata", "artist=Pineal Gateway (procedural)",
                        "-metadata", "comment=" + comment, out_base + ".flac"], check=True)
        log(f"wrote {out_base}.flac (24-bit lossless)")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav,
                        "-c:a", "libmp3lame", "-b:a", "320k", "-joint_stereo", "0", "-id3v2_version", "3",
                        "-metadata", "title=" + title, "-metadata", "artist=Pineal Gateway (procedural)",
                        "-metadata", "comment=" + comment, out_base + ".mp3"], check=True)
        log(f"wrote {out_base}.mp3 (320 kbps, forced L/R stereo)")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log(f"ffmpeg encode skipped: {e}")


DESIGNS = {"full": design_full, "daily": design_daily}

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    if variant not in DESIGNS:
        sys.exit(f"usage: session.py [{'|'.join(DESIGNS)}] [output_base]")
    out = sys.argv[2] if len(sys.argv) > 2 else f"output/pineal_gateway_{variant}"
    render(DESIGNS[variant](), out)
