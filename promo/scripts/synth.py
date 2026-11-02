"""The instruments the soundtracks are played on, all synthesised: pads,
plucks, bass, drums and transitions. Mono numpy arrays at SR."""
import numpy as np
from scipy.signal import butter, sosfilt

SR = 48_000
rng = np.random.default_rng(7)


def hz(note):
    return 440 * 2 ** ((note - 69) / 12)


def times(duration):
    return np.arange(int(duration * SR)) / SR


def filtered(signal, kind, cutoff, order=2):
    return sosfilt(butter(order, cutoff, kind, fs=SR, output="sos"), signal)


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
