"""Simulate 2-speaker reverberant room mixtures with pyroomacoustics.

Unlike the mystery bundled audio/mixture.wav (no known room parameters),
every mixture generated here has a documented, controllable RT60: we pick
a target RT60, solve for the wall absorption + image-source order with
Sabine's formula (pra.inverse_sabine), simulate the room, and then
double-check the RT60 actually achieved from the simulated impulse
response (Schroeder backward integration, pra.experimental.rt60.measure_rt60).

Dry sources: two CMU ARCTIC speakers already bundled in the repo at
online_cpp/samples/ (16 kHz, so no resampling needed). Each speaker's track
loops its own clip continuously (so most of the time both are talking at
once, the continuous spatial overlap that spatial-covariance IVA needs to
hold a stable per-speaker identity), but after each repeat there's a
PAUSE_PROB chance of a silence gap before the next repeat, independently
per speaker -- so occasionally one or both fall silent for a while, giving
some genuinely single-speaker/quiet moments without making the mixture
non-overlapping throughout. (A fully non-overlapping alternating-turns
version was tried and found to break stable per-output-channel identity:
with strictly one source active at a time, the algorithm has no cue to
route a given speaker to the same output channel across turns.)

For each target RT60 writes, into <project>/audio/pyroom/:
    mixture_rt60_<rt60>.wav        2-channel mic signal (both speakers)
    reference_rt60_<rt60>_src1.wav  speaker 1's reverberant image at mic 1
    reference_rt60_<rt60>_src2.wav  speaker 2's reverberant image at mic 1
(references = the corresponding row of room.simulate(return_premix=True),
same "clean-at-mic-1" convention as the bundled demo's reference_src*.wav)

Requires: pip install pyroomacoustics scipy numpy
Run from anywhere, e.g.:
    python tools/make_reverberant_mixtures.py .
"""
import sys
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

PROJECT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT.parent
SAMPLES_DIR = REPO_ROOT / "online_cpp" / "samples"
OUT_DIR = PROJECT / "audio" / "pyroom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 16000
ROOM_DIM = [6.0, 5.0, 3.0]  # metres: a modest meeting room
MIC_CENTER = np.array([3.0, 1.5, 1.5])
MIC_SPACING = 0.08  # 8 cm, 2-element array
SOURCE_DIST = 1.2  # metres from the array center
SOURCE_ANGLES_DEG = [-40.0, 35.0]
NOISE_SNR_DB = 40.0  # mild mic self-noise, applied to the mixture only

SPEAKER_FILES = [
    "cmu_arctic_us_aew_a0001.wav",
    "cmu_arctic_us_axb_a0004.wav",
]

TARGET_RT60S = [0.2, 0.4, 0.6]  # seconds
DURATION_SEC = 60.0  # total length of each speaker's track
PAUSE_PROB = 0.3  # chance of a silence gap after each repeat, per speaker
PAUSE_SEED = 0


def load_speaker(name):
    sr, data = wavfile.read(SAMPLES_DIR / name)
    assert sr == FS, f"{name} is {sr} Hz, expected {FS}"
    return data.astype(np.float64) / 32768.0


def fit_to_length(sig, n):
    if len(sig) >= n:
        return sig[:n]
    out = np.zeros(n)
    out[: len(sig)] = sig
    return out


def build_alternating_tracks(speakers, duration_sec):
    """One track per speaker, all the same length. At any sample at most one
    track is non-zero: speakers take turns, each turn repeating that
    speaker's own clip, round-robin, until every track reaches n_target
    samples (the last turn is truncated, possibly mid-word). Kept here as an
    option, but not used by default -- see module docstring for why."""
    n_target = int(round(duration_sec * FS))
    n_speakers = len(speakers)
    tracks = [[] for _ in range(n_speakers)]
    turn = 0
    while sum(len(chunk) for chunk in tracks[0]) < n_target:
        speaker_idx = turn % n_speakers
        turn_len = len(speakers[speaker_idx])
        for k in range(n_speakers):
            tracks[k].append(speakers[speaker_idx] if k == speaker_idx else np.zeros(turn_len))
        turn += 1
    return [fit_to_length(np.concatenate(chunks), n_target) for chunks in tracks]


def build_mostly_overlapping_tracks(speakers, duration_sec, pause_prob, seed):
    """One track per speaker, all the same length. Each track loops its own
    clip back-to-back, but after every repeat there's `pause_prob` chance of
    a silence gap (same length as that repeat) before continuing. Each
    speaker draws from the same advancing RNG stream, so their pause
    patterns are independent of each other: most of the time every speaker
    is talking (continuous overlap), but occasionally one or more fall
    silent for a stretch."""
    n_target = int(round(duration_sec * FS))
    rng = np.random.default_rng(seed)
    tracks = []
    for sig in speakers:
        chunks = []
        total = 0
        while total < n_target:
            chunks.append(sig)
            total += len(sig)
            if rng.random() < pause_prob:
                gap = np.zeros(len(sig))
                chunks.append(gap)
                total += len(gap)
        tracks.append(fit_to_length(np.concatenate(chunks), n_target))
    return tracks


def mic_positions():
    offset = np.array([MIC_SPACING / 2, 0.0, 0.0])
    return np.stack([MIC_CENTER - offset, MIC_CENTER + offset], axis=1)  # (3, n_mic)


def source_positions():
    pts = []
    for angle_deg in SOURCE_ANGLES_DEG:
        a = np.deg2rad(angle_deg)
        pts.append(MIC_CENTER + SOURCE_DIST * np.array([np.cos(a), np.sin(a), 0.0]))
    return pts


def peak_normalize(x, peak=0.9):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def write_wav(path, sig, n_chan):
    sig = sig.reshape(-1, n_chan) if n_chan > 1 else sig
    pcm = np.clip(np.round(sig * 32768.0), -32768, 32767).astype(np.int16)
    wavfile.write(path, FS, pcm)


def run_one(rt60_target, speakers):
    n_src = len(speakers)

    e_absorption, max_order = pra.inverse_sabine(rt60_target, ROOM_DIM)
    room = pra.ShoeBox(
        ROOM_DIM,
        fs=FS,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    room.add_microphone_array(pra.MicrophoneArray(mic_positions(), FS))
    for sig, pos in zip(speakers, source_positions()):
        room.add_source(pos, signal=sig)

    room.compute_rir()
    measured = [pra.experimental.rt60.measure_rt60(rir, fs=FS) for rir in room.rir[0]]

    premix = room.simulate(return_premix=True)  # (n_src, n_mic, n_samples)
    mix = np.sum(premix, axis=0)  # (n_mic, n_samples)

    rng = np.random.default_rng(0)
    sig_power = np.mean(mix**2)
    noise_power = sig_power / (10 ** (NOISE_SNR_DB / 10))
    mix = mix + rng.normal(scale=np.sqrt(noise_power), size=mix.shape)

    mix = peak_normalize(mix.T)  # (n_samples, n_mic)
    tag = f"rt60_{rt60_target:g}"
    write_wav(OUT_DIR / f"mixture_{tag}.wav", mix, n_chan=n_src)

    for k in range(n_src):
        ref = peak_normalize(premix[k, 0, :])
        write_wav(OUT_DIR / f"reference_{tag}_src{k + 1}.wav", ref, n_chan=1)

    print(
        f"RT60 target={rt60_target:.2f}s  absorption={e_absorption:.3f}  "
        f"max_order={max_order}  measured-per-source={[f'{m:.3f}s' for m in measured]}"
    )


def main():
    dry = [load_speaker(f) for f in SPEAKER_FILES]
    speakers = build_mostly_overlapping_tracks(dry, DURATION_SEC, PAUSE_PROB, PAUSE_SEED)
    print(f"Room {ROOM_DIM} m, {len(speakers)} sources at {SOURCE_ANGLES_DEG} deg, "
          f"{SOURCE_DIST} m from a {MIC_SPACING * 100:.0f} cm 2-mic array, "
          f"mostly overlapping with pause_prob={PAUSE_PROB} over {DURATION_SEC:.0f}s\n")
    for rt60 in TARGET_RT60S:
        run_one(rt60, speakers)
    print(f"\nWrote mixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
