from collections import defaultdict

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


def pack_hash(f1, f2, delta_t):
	return (int(f1) << 22) | (int(f2) << 12) | int(delta_t)


def generate_hashes_from_peaks(
	final_peaks,
	dt_min=5,
	dt_max=50,
	fan_value=10,
):
	"""Generate packed hashes from (freq_bin, time_frame) peaks."""
	peaks_sorted = np.asarray(final_peaks, dtype=np.int32)
	if peaks_sorted.size == 0:
		return np.empty((0, 2), dtype=np.int64)

	if peaks_sorted.ndim != 2 or peaks_sorted.shape[1] != 2:
		raise ValueError("final_peaks must have shape (N, 2) with (freq, time)")

	order = np.argsort(peaks_sorted[:, 1], kind="mergesort")
	peaks_sorted = peaks_sorted[order]

	freqs = peaks_sorted[:, 0]
	times = peaks_sorted[:, 1]
	n = len(peaks_sorted)

	starts = np.searchsorted(times, times + dt_min, side="left")
	ends = np.searchsorted(times, times + dt_max, side="right")

	total = 0
	for i in range(n):
		start = max(starts[i], i + 1)
		end = min(ends[i], start + fan_value)
		if start < end:
			total += end - start

	hashes = np.empty((total, 2), dtype=np.int64)

	ptr = 0
	for i in range(n):
		start = max(starts[i], i + 1)
		end = min(ends[i], start + fan_value)
		if start >= end:
			continue

		size = end - start
		target_freqs = freqs[start:end].astype(np.int64, copy=False)
		delta_t = (times[start:end] - times[i]).astype(np.int64, copy=False)

		packed = (np.int64(freqs[i]) << 22) | (target_freqs << 12) | delta_t

		hashes[ptr : ptr + size, 0] = packed
		hashes[ptr : ptr + size, 1] = times[i]
		ptr += size

	return hashes


def generate_hashes_for_audio(
	audio_path,
	*,
	sr=22050,
	n_fft=2048,
	hop_length=512,
	percentile=80,
	neighborhood_size=20,
	activity_percentile=20,
	floor_percentile=10,
	min_floor_db=-60,
	max_per_time=4,
	dt_min=5,
	dt_max=50,
	fan_value=10,
):
	y, _ = librosa.load(audio_path, sr=sr)
	max_abs = np.max(np.abs(y))
	if max_abs > 0:
		y = y / max_abs

	S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False))
	S_db = librosa.amplitude_to_db(S, ref=np.max)

	local_max = maximum_filter(S_db, size=neighborhood_size) == S_db
	thresholds = np.percentile(S_db, percentile, axis=0)

	frame_energy = np.mean(S_db, axis=0)
	active_frames = frame_energy >= np.percentile(frame_energy, activity_percentile)
	adaptive_floor = np.percentile(S_db, floor_percentile)
	abs_floor_db = max(adaptive_floor, min_floor_db)

	detected = (
		local_max
		& (S_db >= thresholds[np.newaxis, :])
		& (S_db >= abs_floor_db)
		& active_frames[np.newaxis, :]
	)

	peaks = np.argwhere(detected)
	if peaks.size == 0:
		return np.empty((0, 2), dtype=np.int64)

	buckets = defaultdict(list)
	for f, t in peaks:
		buckets[t].append((f, t, S_db[f, t]))

	final_peaks = []
	for t in buckets:
		selected = sorted(buckets[t], key=lambda x: -x[2])[:max_per_time]
		final_peaks.extend((f, t) for f, t, _ in selected)

	final_peaks = np.array(final_peaks, dtype=np.int32)

	return generate_hashes_from_peaks(
		final_peaks,
		dt_min=dt_min,
		dt_max=dt_max,
		fan_value=fan_value,
	)


def generate_hashes_for_audio_files(audio_paths, **kwargs):
	return {path: generate_hashes_for_audio(path, **kwargs) for path in audio_paths}
