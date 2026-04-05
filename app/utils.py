import os
import time
from collections import Counter, defaultdict
import librosa
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from scipy.ndimage import maximum_filter
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def generate_hashes_from_peaks(
	final_peaks,
	dt_min=5,
	dt_max=50,
	fan_value=10,
	df_offset=2048,
):
	peaks_sorted = np.asarray(final_peaks, dtype=np.int32)
	if peaks_sorted.size == 0:
		return np.empty((0, 3), dtype=np.int64)

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

	hashes = np.empty((total, 3), dtype=np.int64)

	ptr = 0
	for i in range(n):
		start = max(starts[i], i + 1)
		end = min(ends[i], start + fan_value)
		if start >= end:
			continue

		size = end - start
		target_freqs = freqs[start:end].astype(np.int64, copy=False)
		delta_t = (times[start:end] - times[i]).astype(np.int64, copy=False)
		delta_f = target_freqs - np.int64(freqs[i])
		encoded_df = np.clip(delta_f + np.int64(df_offset), 0, 4095)
		packed = (encoded_df << 12) | delta_t

		hashes[ptr : ptr + size, 0] = packed
		hashes[ptr : ptr + size, 1] = times[i]
		hashes[ptr : ptr + size, 2] = freqs[i]
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
	df_offset=2048,
):
	y, _ = librosa.load(audio_path, sr=sr, duration=5)
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
		df_offset=df_offset,
	)


def generate_hashes_for_audio_files(audio_paths, **kwargs):
	return {path: generate_hashes_for_audio(path, **kwargs) for path in audio_paths}


def ingest_song_from_audio(
	audio_path,
	*,
	title=None,
	artist=None,
	hash_kwargs=None,
	page_size=1000,
):
	hash_kwargs = hash_kwargs or {}
	hashes = generate_hashes_for_audio(audio_path, **hash_kwargs)

	if title is None:
		title = os.path.splitext(os.path.basename(audio_path))[0]

	duration = float(librosa.get_duration(path=audio_path))

	conn = psycopg2.connect(DATABASE_URL)
	cur = conn.cursor()
	try:
		cur.execute(
			"""
			INSERT INTO songs (title, artist, duration)
			VALUES (%s, %s, %s)
			RETURNING id;
			""",
			(title, artist, duration),
		)
		song_id = cur.fetchone()[0]

		if hashes.size > 0:
			data = [
				(int(h), int(song_id), int(offset), int(anchor_freq))
				for h, offset, anchor_freq in hashes
			]
			execute_values(
				cur,
				"""
				INSERT INTO fingerprints (hash, song_id, time_offset, anchor_freq)
				VALUES %s
				""",
				data,
				page_size=page_size,
			)

		conn.commit()
	finally:
		cur.close()
		conn.close()

	return {
		"song_id": int(song_id),
		"title": title,
		"artist": artist,
		"duration": float(duration),
		"hash_count": int(len(hashes)),
	}


def match_audio_hashes(
	query_hashes,
	*,
	anchor_tol=2,
	min_score=10,
):
	start_time = time.perf_counter()

	if query_hashes.size == 0:
		elapsed_s = round(time.perf_counter() - start_time, 4)
		return {
			"song_id": None,
			"title": None,
			"artist": None,
			"score": 0,
			"hash_count": 0,
			"anchor_tol": int(anchor_tol),
			"min_score": int(min_score),
			"time_taken_s": elapsed_s,
		}

	query_hash_list = query_hashes[:, 0].astype(int).tolist()

	conn = psycopg2.connect(DATABASE_URL)
	cur = conn.cursor()
	try:
		cur.execute(
			"""
			SELECT hash, song_id, time_offset, anchor_freq
			FROM fingerprints
			WHERE hash = ANY(%s)
			""",
			(query_hash_list,),
		)
		rows = cur.fetchall()
	finally:
		cur.close()
		conn.close()

	db_map = defaultdict(list)
	for h, song_id, offset, anchor_freq in rows:
		db_map[int(h)].append((int(song_id), int(offset), int(anchor_freq)))

	votes = defaultdict(list)
	for h, q_offset, q_anchor_freq in query_hashes:
		h = int(h)
		q_offset = int(q_offset)
		q_anchor_freq = int(q_anchor_freq)

		for song_id, db_offset, db_anchor_freq in db_map.get(h, []):
			if abs(db_anchor_freq - q_anchor_freq) > anchor_tol:
				continue
			votes[song_id].append(db_offset - q_offset)

	best_song = None
	best_score = 0
	for song_id, deltas in votes.items():
		count = Counter(deltas).most_common(1)[0][1]
		if count > best_score:
			best_score = count
			best_song = song_id

	if best_score < min_score:
		best_song = None

	best_title = None
	best_artist = None
	if best_song is not None:
		conn = psycopg2.connect(DATABASE_URL)
		cur = conn.cursor()
		try:
			cur.execute("SELECT title, artist FROM songs WHERE id = %s", (best_song,))
			row = cur.fetchone()
			if row:
				best_title = row[0]
				best_artist = row[1]
		finally:
			cur.close()
			conn.close()

	elapsed_s = round(time.perf_counter() - start_time, 4)
	return {
		"song_id": int(best_song) if best_song is not None else None,
		"title": best_title,
		"artist": best_artist,
		"score": int(best_score) if best_song is not None else 0,
		"hash_count": int(len(query_hashes)),
		"anchor_tol": int(anchor_tol),
		"min_score": int(min_score),
		"time_taken_s": elapsed_s,
	}


def search_song_by_audio_path(
	audio_path,
	*,
	anchor_tol=2,
	min_score=10,
	hash_kwargs=None,
):
	hash_kwargs = hash_kwargs or {}
	query_hashes = generate_hashes_for_audio(audio_path, **hash_kwargs)
	return match_audio_hashes(
		query_hashes,
		anchor_tol=anchor_tol,
		min_score=min_score,
	)