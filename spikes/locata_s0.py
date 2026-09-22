"""S0 spike: one DOA estimate from one real LOCATA recording.

Throwaway code; the findings are the deliverable (see spikes/README.md). Dataset: LOCATA dev
set, Task 1 (single static loudspeaker, static array), recording 1, DICIT array. Fetch the
files first with ``python spikes/fetch_locata.py``.

Method: STFT -> voice-active frames -> per-bin sample covariance on a uniform linear
subarray -> narrowband MUSIC (M = 1) per bin -> incoherent fusion (mean of peak-normalised
pseudospectra) over the subarray's alias-free band.
"""

import warnings
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from doabench.arrays import SensorArray
from doabench.estimators import music_spectrum
from doabench.grids import uniform_grid

ROOT = Path(__file__).resolve().parents[1] / "data/external/locata/dev/task1/recording1/dicit"
C = 343.0  # speed of sound, m/s
GRID = uniform_grid(-90, 90, 0.25)


def read_positions():
    """Microphone and source positions in the array's local frame (metres)."""
    arr = np.genfromtxt(ROOT / "position_array_dicit.txt", names=True, delimiter="\t")
    src = np.genfromtxt(ROOT / "position_source_loudspeaker1.txt", names=True, delimiter="\t")
    centre = np.array([arr[a][0] for a in "xyz"])
    rot = np.array([[arr[f"rotation_{i}{j}"][0] for j in (1, 2, 3)] for i in (1, 2, 3)])
    mics = np.array([[arr[f"mic{k}_{a}"][0] for a in "xyz"] for k in range(1, 16)])
    source = np.array([src[a][0] for a in "xyz"])
    # global = centre + R @ local  (this is the convention that makes the array collinear)
    return (mics - centre) @ rot, (source - centre) @ rot


def uniform_subarray(mics_local, spacing):
    """Indices of the longest run of mics at x = k * spacing, k = 0, ±1, ±2, ... (z = 0)."""
    on_axis = {round(p[0] / spacing * 1e6): i for i, p in enumerate(mics_local) if abs(p[2]) < 1e-6}
    members = [on_axis[0]]
    for sign in (1, -1):
        k = sign
        while round(k * 1e6) in on_axis:
            members.append(on_axis[round(k * 1e6)])
            k += sign
    return sorted(members, key=lambda i: mics_local[i, 0])


def main():
    mics_local, src_local = read_positions()
    dist = np.linalg.norm(src_local)
    u = src_local / dist
    # A linear array along x measures only the cone angle arcsin(u_x): it cannot tell
    # front from back, or separate azimuth from elevation.
    truth = np.degrees(np.arcsin(u[0]))
    print(
        f"source in array frame: distance {dist:.2f} m, "
        f"elevation {np.degrees(np.arcsin(u[2])):.1f} deg"
    )
    print(f"ground-truth cone angle (broadside = 0): {truth:.2f} deg\n")

    with warnings.catch_warnings():  # LOCATA WAVs carry a non-standard chunk scipy skips
        warnings.simplefilter("ignore", scipy.io.wavfile.WavFileWarning)
        fs, audio = scipy.io.wavfile.read(ROOT / "audio_array_dicit.wav")
    vad = np.loadtxt(ROOT / "VAD_dicit_loudspeaker1.txt", skiprows=1) > 0.5
    stft = ShortTimeFFT(hann(1024, sym=False), hop=512, fs=fs)
    spec = stft.stft(audio.T)  # (mics, bins, frames)
    centre_sample = (stft.p_min + np.arange(spec.shape[2])) * stft.hop
    active = vad[np.clip(centre_sample, 0, vad.size - 1)]
    print(
        f"{audio.shape[1]} ch @ {fs} Hz, {audio.shape[0] / fs:.2f} s, dtype {audio.dtype}; "
        f"{active.sum()}/{active.size} STFT frames voice-active\n"
    )

    print(
        f"{'spacing':>8} {'mics':>4} {'band (Hz)':>11} {'far field':>9} {'MUSIC':>8} {'error':>7}"
    )
    for spacing in (0.04, 0.08, 0.16, 0.32):
        members = uniform_subarray(mics_local, spacing)
        pos_m = mics_local[members] - mics_local[members].mean(0)
        f_hi = C / (2 * spacing)  # spatial aliasing above this
        f_lo = max(200.0, f_hi / 8)
        fraunhofer = 2 * np.ptp(pos_m[:, 0]) ** 2 * f_hi / C  # 2D²/λ at the top of the band
        fused = np.zeros_like(GRID)
        for b in np.flatnonzero((stft.f >= f_lo) & (stft.f <= f_hi)):
            xb = spec[members][:, b, active]
            r = xb @ xb.conj().T / xb.shape[1]
            p = music_spectrum(r, SensorArray(pos_m * stft.f[b] / C), 1, GRID)
            fused += p / p.max()
        est = GRID[np.argmax(fused)]
        print(
            f"{spacing * 100:6.0f}cm {len(members):4d} {f_lo:5.0f}-{f_hi:<5.0f} "
            f"{'yes' if dist > fraunhofer else 'NO':>9} {est:8.2f} {est - truth:7.2f}"
        )


if __name__ == "__main__":
    main()
