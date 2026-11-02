"""A 16-second music loop for the App Store's header and search results
videos, which play on repeat (and start muted, so it is only texture).

Eight bars at 120 BPM, the trailer's chords twice. Everything wraps
around: notes that ring past the end, the echoes and the reverb tail
continue at the start, so the loop has no seam. Soft throughout, as Apple
asks: no impacts, no claps.

    python3 scripts/loop.py public/loop.wav      (needs numpy and scipy)
"""
import sys
import wave

import numpy as np

from synth import SR, bass, filtered, hat, hz, kick, pad_note, pluck, rng, times

BEAT = 0.5
BAR = 4 * BEAT
BARS = 8
N = int(SR * BAR * BARS)
CHORDS = [[53, 57, 60, 64], [52, 55, 60, 62], [50, 53, 57, 60], [46, 50, 53, 57]]
ROOTS = [41, 40, 38, 34]


def stereo():
    return np.zeros((2, N))


def add(bus, signal, at, pan=0.0, gain=1.0):
    """Mixes a mono signal in at `at` seconds, wrapping past the end."""
    index = (int(at * SR) + np.arange(len(signal))) % N
    left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    np.add.at(bus[0], index, signal * gain * left * np.sqrt(2))
    np.add.at(bus[1], index, signal * gain * right * np.sqrt(2))


pads, arp, drums, low = stereo(), stereo(), stereo(), stereo()
kicks = []
for bar in range(BARS):
    notes = CHORDS[bar % 4]
    t0 = bar * BAR
    for i, note in enumerate(notes):
        add(pads, pad_note(hz(note), BAR), t0, pan=(i - 1.5) * 0.35)
    for k in range(8):
        note = notes[[0, 1, 2, 3, 2, 1, 2, 3][k]] + 12
        add(arp, pluck(hz(note), decay=0.18), t0 + k * BEAT / 2, pan=0.3 * np.sin(k), gain=0.13 * (1 if k % 4 == 0 else 0.7))
    for beat in range(4):
        at = t0 + beat * BEAT
        kicks.append(at)
        add(drums, kick(), at, gain=0.45)
        add(drums, hat(), at + BEAT / 2, pan=0.3, gain=0.28)
    for eighth in range(8):
        add(low, bass(hz(ROOTS[bar % 4]), BEAT / 2 * 0.9), t0 + eighth * BEAT / 2, gain=0.2)

# A dotted-eighth echo on the arpeggio, wrapped like everything else.
mono = arp.mean(axis=0)
for i, (delay, gain) in enumerate(((0.375, 0.35), (0.75, 0.22), (1.125, 0.13))):
    add(arp, mono, delay, pan=-0.8 if i % 2 == 0 else 0.8, gain=gain)

# Gentle sidechain under the kicks.
duck = np.ones(N)
t = np.arange(N) / SR
for at in kicks:
    since = (t - at) % (BAR * BARS)
    duck *= 1 - 0.35 * np.exp(-since / 0.11) * (since < 0.45)
pads = np.vstack([filtered(ch, "lowpass", 2000) for ch in pads]) * duck
low *= duck

# Reverb by circular convolution, so the tail of the last bar rings into
# the first.
ir_t = times(2.6)
ir = np.vstack([filtered(rng.standard_normal(len(ir_t)), "lowpass", 6500) * np.exp(-ir_t / 0.65) for _ in range(2)])
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
send = pads * 0.6 + arp * 0.5
wet = np.vstack([np.real(np.fft.ifft(np.fft.fft(send[c]) * np.fft.fft(ir[c], N))) for c in range(2)])

mix = pads + arp + drums + low + 0.55 * wet
mix = np.tanh(1.2 * mix / np.abs(mix).max()) / np.tanh(1.2) * 10 ** (-3 / 20)

out = sys.argv[1] if len(sys.argv) > 1 else "public/loop.wav"
with wave.open(out, "wb") as f:
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(SR)
    f.writeframes((mix.T * 32767).astype("<i2").tobytes())
seam = np.abs(mix[:, 0] - mix[:, -1]).max()
print(f"{out}: {N / SR:.0f} s loop, peak {np.abs(mix).max():.2f}, jump at the seam {seam:.4f}")
