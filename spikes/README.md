# S0 spike: first DOA estimate from real LOCATA data

Throwaway code, kept for the record. The purpose was to hit dataset and convention surprises
in sprint 0 rather than sprint 5. The findings below are the deliverable.

```sh
uv run python spikes/fetch_locata.py   # ~30 MB, not the 6.2 GB archive
uv run python spikes/locata_s0.py
```

## Result

LOCATA dev set, Task 1 (one static loudspeaker, static array), recording 1, DICIT array.
The source is 2.41 m away at 5.7° elevation. Ground-truth cone angle: **−57.5°**.

| uniform subarray | mics | band (Hz) | far field? | MUSIC | Bartlett | MUSIC error |
|---|---:|---|---|---:|---:|---:|
| 4 cm | 5 | 536–4288 | yes | −70.3° | — | −12.7° |
| 8 cm | 5 | 268–2144 | yes | −53.0° | −53.8° | +4.5° |
| 16 cm | 5 | 200–1072 | no | −52.3° | −52.5° | +5.3° |
| 32 cm | 7 | 200–536 | no | −53.5° | −52.8° | +4.0° |

Method: 1024-point STFT with 50% overlap, voice-active frames only (208 of 304), per-bin
MUSIC with M = 1, incoherent fusion of peak-normalised pseudospectra over each subarray's
alias-free band, c = 343 m/s. The 4 cm subarray has only a 16 cm aperture, so its beam is
several times wider than the others'. A poor estimate from it is expected.

## Findings

**Access.** dev.zip is 6.2 GB and eval.zip 13 GB (Zenodo record 3630471). Zenodo honours
HTTP range requests, so `fetch_locata.py` reads the zip's central directory remotely and
pulls one recording on one array (29 MB, 7 requests). The licence is ODC-By 1.0:
redistribution is allowed with attribution. The data still stays out of git; the fetch
script is the interface.

**Layout.** `dev/taskN/recordingK/<array>/` holds 10 files per array: array audio, source
audio, array and source position tracks, per-sample voice activity (VAD), per-sample
timestamps, and `required_time.txt`. Task 1 recording 1 has three arrays: `benchmark2`
(robot head), `dicit` and `eigenmike`.

**Audio.** 48 kHz, stored as **float64** WAV with a non-standard chunk; scipy warns and then
reads it correctly. DICIT has 15 channels. This recording is 3.23 s long.

**Positions.** Tab-separated, in metres, in a global room frame, one row every ~9 ms. Each
row gives the array centre c, a rotation R and every microphone position. The convention is
**global = c + R·local**, verified because only that orientation makes DICIT exactly
collinear. In Task 1 everything is static to within 1e-14 m. Audio timestamps start 4 ms
after the first position row.

**DICIT geometry** (array frame, cm): 13 collinear mics at 0, ±4, ±8, ±16, ±32, ±64 and ±96,
plus two raised mics at (±96, 0, 32). It is a *nested* array, not a ULA, but it contains
uniform subarrays at 4, 8 and 16 cm spacing (5 mics each) and at 32 cm (7 mics).

**What a linear array can observe.** Only the cone angle arcsin(uₓ). Front and back are
ambiguous, and elevation folds into the estimate, so the ground truth has to be defined
that way. A planar or 3-D array is needed for azimuth and elevation separately.

## Open question for S5: a ~4.5° bias toward broadside

The three larger subarrays all land 4–5° short of the truth, and conventional beamforming
agrees with MUSIC to within about 1°. So the bias is in the data or the model, not the
estimator. Candidate causes:

- **Speed of sound: ruled out.** Pulling −57.5° to −53° needs c ≈ 362 m/s, about 54 °C.
- **Tracked position versus acoustic centre.** Would need about 19 cm of lateral offset at
  2.41 m. That's plausible only if the tracker marker is well away from the driver.
- **Early reflections** pulling the fused spectrum.
- **Near field.** It can't explain the 8 cm subarray, which is far-field. It may add to the
  error for the 16 and 32 cm subarrays at 2.4 m.

The discriminating test: a bias that is consistent across Task 1 recordings points to
geometry or ground truth; one that varies from recording to recording points to room
acoustics.

## Design consequences for the package

1. **`SensorArray` should store metres, not wavelengths.** Wavelength units forced a new
   array object per frequency bin. From S4 on, steering should take frequency and c.
2. **Steering needs azimuth/elevation (3-D)**, plus a cone-angle helper for linear
   subarrays.
3. **ULA-only methods need subarray selection.** Root-MUSIC, ESPRIT and spatial smoothing
   will run on DICIT's uniform subarrays, and `uniform_subarray()` here is a first
   version.
4. **Near-field is a live issue.** The 16 and 32 cm subarrays violate the Fraunhofer
   condition at this distance. S5 needs either a spherical-wavefront steering option or
   a restriction to far-field subarrays.
