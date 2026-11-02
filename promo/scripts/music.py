"""The soundtrack, made from scratch here so it's ours to use anywhere.

33 s at 120 BPM: a beat is 0.5 s (15 video frames), a bar 2 s, and the
trailer cuts on bars, so the music follows the pictures:

    0-4 s    the opening words, one pluck per word, rising into
    4 s      the logo: impact, the chords begin
    8-28 s   the features: drums, bass, arpeggio; a whoosh on every cut
    28-30 s  privacy: drums drop out, a riser
    30-33 s  the end card: impact and a last chord, fading out

    python3 scripts/music.py public/music.wav      (needs numpy and scipy)

also writes public/music-bed.wav, the same without the whooshes.
"""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48_000
BEAT = 0.5
BAR = 4 * BEAT
LENGTH = 33.0
N = int(SR * LENGTH)
rng = np.random.default_rng(7)


def hz(note):
    return 440 * 2 ** ((note - 69) / 12)


# One chord a bar. The opening leans on D minor and B flat, then the
# features loop F major 7, C over E, D minor 7, B flat major 7.
INTRO = [[50, 53, 57, 60], [46, 50, 53, 57]]
LOOP = [[53, 57, 60, 64], [52, 55, 60, 62], [50, 53, 57, 60], [46, 50, 53, 57]]
ROOTS = {53: 41, 52: 40, 50: 38, 46: 34}


def chord(bar):
    if bar < 2:
        return INTRO[bar]
    if bar == 14:
        return LOOP[3]                      # B flat under privacy...
    if bar >= 15:
        return [41, 48, 57, 60, 64, 67]     # ...home to F for the end card
    return LOOP[(bar - 2) % 4]


def stereo():
    return np.zeros((2, N))


def add(bus, signal, at, pan=0.0, gain=1.0):
    """Mixes a mono signal into a stereo bus at `at` seconds."""
    start = int(at * SR)
    if start >= N:
        return
    signal = signal[: N - start] * gain
    left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    bus[0, start:start + len(signal)] += signal * left * np.sqrt(2)
    bus[1, start:start + len(signal)] += signal * right * np.sqrt(2)


def times(duration):
    return np.arange(int(duration * SR)) / SR


def filtered(signal, kind, cutoff, order=2):
    return sosfilt(butter(order, cutoff, kind, fs=SR, output="sos"), signal)


# --- Instruments ---------------------------------------------------------

def pad_note(freq, duration, attack=0.35, release=0.7):
    t = times(duration + release)
    out = np.zeros_like(t)
    for cents, phase in ((-8, 0.3), (0, 1.7), (8, 4.1)):
        f = freq * 2 ** (cents / 1200)
        for h in range(1, 11):
            out += np.sin(2 * np.pi * f * h * t + phase * h) / h
    env = np.minimum(t / attack, 1) * np.clip((duration + release - t) / release, 0, 1)
    return out * env * 0.05


def pluck(freq, decay=0.22):
    t = times(1.2)
    tone = (np.sin(2 * np.pi * freq * t)
            + 0.45 * np.sin(4 * np.pi * freq * t) * np.exp(-t / 0.09)
            + 0.18 * np.sin(2 * np.pi * freq * 3.01 * t) * np.exp(-t / 0.05))
    return tone * np.exp(-t / decay) * (1 - np.exp(-t / 0.002))


def bass(freq, duration):
    t = times(duration)
    tone = np.tanh(1.8 * (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)))
    return tone * np.exp(-t / 0.3) * np.clip((duration - t) / 0.02, 0, 1) * (1 - np.exp(-t / 0.004))


def kick():
    t = times(0.5)
    phase = 2 * np.pi * np.cumsum(46 + 95 * np.exp(-t / 0.035)) / SR
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.003) * 0.25
    return np.sin(phase) * np.exp(-t / 0.26) + click


def clap():
    t = times(0.35)
    noise = filtered(rng.standard_normal(len(t)), "bandpass", [900, 3500])
    env = np.exp(-t / 0.11)
    for burst in (0.0, 0.011, 0.022):
        env = env + 1.4 * np.exp(-np.maximum(t - burst, 0) / 0.006) * (t >= burst)
    return noise * env * 0.35


def hat(open_=False):
    t = times(0.4)
    noise = filtered(rng.standard_normal(len(t)), "highpass", 7000)
    return noise * np.exp(-t / (0.09 if open_ else 0.028)) * 0.5


def whoosh(length=0.7):
    """Rises into a cut, so it is placed `length` seconds before one."""
    t = times(length + 0.25)
    noise = filtered(rng.standard_normal(len(t)), "bandpass", [600, 6000])
    rise = np.where(t < length, (t / length) ** 3, np.exp(-(t - length) / 0.08))
    return noise * rise * 0.22


def riser(length):
    t = times(length)
    sweep = np.sin(2 * np.pi * np.cumsum(180 + 900 * (t / length) ** 2) / SR)
    noise = filtered(rng.standard_normal(len(t)), "highpass", 1500)
    return (sweep * 0.12 + noise * 0.18) * (t / length) ** 2.5


def impact():
    t = times(2.5)
    sub = np.sin(2 * np.pi * np.cumsum(32 + 40 * np.exp(-t / 0.25)) / SR) * np.exp(-t / 0.9)
    wash = filtered(rng.standard_normal(len(t)), "lowpass", 4000) * np.exp(-t / 0.5) * 0.35
    return sub * 0.9 + wash


# --- Arrangement ---------------------------------------------------------

dry, send, pads, low, cuts = stereo(), stereo(), stereo(), stereo(), stereo()
kicks = []

# Pads, every bar; quiet under the opening words, full from the logo.
for bar in range(16):
    level = 0.45 if bar < 2 else (0.6 if bar == 14 else 1.0)
    for i, note in enumerate(chord(bar)):
        add(pads, pad_note(hz(note), BAR if bar < 15 else 3.0), bar * BAR, pan=(i - 1.5) * 0.35, gain=level)

# The opening words: one pluck each on the first four beats, then
# "No cloud." and "Just your iPhone." on beats 1 and 3 of the next bar.
for at, note in ((0.0, 69), (0.5, 72), (1.0, 74), (1.5, 77), (2.0, 74), (3.0, 77)):
    p = pluck(hz(note), decay=0.45)
    add(dry, p, at, gain=0.5)
    add(send, p, at, gain=0.5)
add(dry, riser(2.0), 2.0, gain=0.8)
add(send, riser(2.0), 2.0, gain=0.4)

# Impacts on the logo and the end card.
for at in (4.0, 30.0):
    add(dry, impact(), at, gain=0.9)
    add(send, impact(), at, gain=0.5)

# Arpeggio: eighths under the logo, sixteenths through the features,
# filtered back during the privacy bar. A dotted-eighth echo, ping-pong.
pattern = [0, 1, 2, 3, 2, 1, 2, 3]
arp = stereo()
for bar in range(2, 15):
    notes = chord(bar)
    step = BEAT / 2 if bar < 4 else BEAT / 4
    count = int(BAR / step)
    for k in range(count):
        note = notes[pattern[k % len(pattern)]] + 12 + (12 if bar >= 8 and k % 8 == 7 else 0)
        accent = 1.0 if k % 4 == 0 else 0.7
        add(arp, pluck(hz(note), decay=0.16), bar * BAR + k * step, pan=0.25 * np.sin(k), gain=0.16 * accent)
mono_arp = arp.mean(axis=0)
echo = stereo()
for i, (delay, gain) in enumerate(((0.375, 0.35), (0.75, 0.22), (1.125, 0.13))):
    add(echo, mono_arp, delay, pan=(-0.8 if i % 2 == 0 else 0.8), gain=gain)
arp_bus = arp + echo
cut = int(28 * SR)
arp_bus[:, cut:] = np.vstack([filtered(ch[cut:], "lowpass", 1200) for ch in arp_bus]) * 0.8
dry += arp_bus
send += arp_bus * 0.5

# Drums and bass through the features.
for bar in range(4, 14):
    t0 = bar * BAR
    root = ROOTS[chord(bar)[0]]
    for beat in range(4):
        at = t0 + beat * BEAT
        kicks.append(at)
        add(dry, kick(), at, gain=0.85)
        if beat in (1, 3):
            add(dry, clap(), at, gain=0.7)
            add(send, clap(), at, gain=0.35)
        add(dry, hat(open_=beat == 3), at + BEAT / 2, pan=0.3, gain=0.5)
        for sixteenth in (0.25, 0.75):
            add(dry, hat(), at + sixteenth * BEAT, pan=-0.3, gain=0.18)
    for eighth in range(8):
        note = root + (12 if eighth % 4 == 3 else 0)
        add(low, bass(hz(note), BEAT / 2 * 0.9), t0 + eighth * BEAT / 2, gain=0.32)

# A whoosh into every cut between features, and into privacy.
for cut_at in (8.0, 12.0, 16.0, 20.0, 24.0, 28.0):
    add(cuts, whoosh(), cut_at - 0.7, gain=0.9)
add(dry, riser(2.0), 28.0, gain=0.9)
add(send, riser(2.0), 28.0, gain=0.4)

# Sidechain: pads and bass duck under each kick.
duck = np.ones(N)
t = np.arange(N) / SR
for at in kicks:
    start = int(at * SR)
    span = t[start:start + int(0.45 * SR)] - at
    duck[start:start + len(span)] *= 1 - 0.55 * np.exp(-span / 0.11)
pads *= duck
low *= duck

pads = np.vstack([filtered(ch, "lowpass", 2200) for ch in pads])
dry += pads + low
send += pads * 0.6

# Reverb: a stereo tail of decaying noise.
ir_t = times(2.6)
ir = np.vstack([filtered(rng.standard_normal(len(ir_t)), "lowpass", 6500) * np.exp(-ir_t / 0.65)
                for _ in range(2)])
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
wet = np.vstack([fftconvolve(send[c], ir[c])[:N] for c in range(2)])
mix = dry + 0.55 * wet

# Master: gentle saturation, -1 dB peak, quick fade in, fade out at the end.
# Written twice: with the whooshes for the trailer, and without them as
# the bed for the short clips, which don't cut where the trailer does.
peak = np.abs(mix + cuts).max()
fade = np.clip(t / 0.02, 0, 1) * np.clip((LENGTH - t) / 2.2, 0, 1)


def master(signal):
    return np.tanh(1.3 * signal / peak) / np.tanh(1.3) * fade * 10 ** (-1 / 20)


def write(path, signal):
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((signal.T * 32767).astype("<i2").tobytes())
    print(f"{path}: {LENGTH:.0f} s, peak {np.abs(signal).max():.2f}")


out = sys.argv[1] if len(sys.argv) > 1 else "public/music.wav"
write(out, master(mix + cuts))
write(out.replace(".wav", "-bed.wav"), master(mix))
