import os
import time
from collections import Counter, defaultdict
import librosa
import numpy as np
from psycopg2.extras import execute_values
from scipy.ndimage import maximum_filter
from app.db import get_connection, release_connection
import pickle
import redis

DATABASE_URL = os.getenv("DATABASE_URL")
redis_client = redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=False,
)

def generate_hashes_from_peaks(
    final_peaks,
    dt_min=5,
    dt_max=50,
    fan_value=5,
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
        hashes[ptr: ptr + size, 0] = packed
        hashes[ptr: ptr + size, 1] = times[i]
        hashes[ptr: ptr + size, 2] = freqs[i]
        ptr += size

    return hashes


def generate_hashes_for_audio(
    audio_path,
    *,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    duration=5,
    percentile=80,
    neighborhood_size=20,
    activity_percentile=20,
    floor_percentile=10,
    min_floor_db=-60,
    max_per_time=3,
    dt_min=5,
    dt_max=50,
    fan_value=5,
    df_offset=2048,
):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)

    max_abs = np.max(np.abs(y))

    if max_abs > 0:
        y = y / max_abs

    S = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
        )
    )

    S_db = librosa.amplitude_to_db(S, ref=np.max)

    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db

    thresholds = np.percentile(S_db, percentile, axis=0)

    frame_energy = np.mean(S_db, axis=0)

    active_frames = (
        frame_energy >= np.percentile(frame_energy, activity_percentile)
    )

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
        return (
        np.empty((0, 3), dtype=np.int64),
        len(y) / sr,
    )

    buckets = defaultdict(list)

    for f, t in peaks:
        buckets[t].append((f, t, S_db[f, t]))

    final_peaks = []

    for t in buckets:
        selected = sorted(
            buckets[t],
            key=lambda x: -x[2]
        )[:max_per_time]

        final_peaks.extend((f, t) for f, t, _ in selected)

    final_peaks = np.array(final_peaks, dtype=np.int32)

    hashes =  generate_hashes_from_peaks(
        final_peaks,
        dt_min=dt_min,
        dt_max=dt_max,
        fan_value=fan_value,
        df_offset=df_offset,
    )
    duration_seconds = len(y) / sr
    return hashes, duration_seconds


def generate_hashes_for_audio_files(audio_paths, **kwargs):
    return {
        path: generate_hashes_for_audio(path, **kwargs)
        for path in audio_paths
	}

def ingest_song_from_audio(
    audio_path,
    *,
    title=None,
    artist=None,
    hash_kwargs=None,
    page_size=1000,
):
    hash_kwargs = hash_kwargs or {}
    hashes, duration = generate_hashes_for_audio(
        audio_path,
        duration=None,
        **hash_kwargs,
    )
    if title is None:
        title = os.path.splitext(
            os.path.basename(audio_path)
        )[0]
    conn = get_connection()
    cur = conn.cursor()
    try:

        cur.execute(
            """
            SELECT id
            FROM songs
            WHERE title = %s
            """,
            (title,),
        )

        if cur.fetchone() is not None:
            raise ValueError(
                f"A song with title '{title}' already exists"
            )
        cur.execute(
            """
            INSERT INTO songs
            (
                title,
                artist,
                duration
            )
            VALUES (%s,%s,%s)
            RETURNING id
            """,
            (
                title,
                artist,
                duration,
            ),
        )

        song_id = cur.fetchone()[0]
        if len(hashes):

            rows = [
                (
                    int(h),
                    int(song_id),
                    int(offset),
                    int(anchor_freq),
                )
                for h, offset, anchor_freq in hashes
            ]
            execute_values(
                cur,
                """
                INSERT INTO fingerprints
                (
                    hash,
                    song_id,
                    time_offset,
                    anchor_freq
                )
                VALUES %s
                """,
                rows,
                page_size=page_size,
            )
        conn.commit()
        # redis_client.flushdb()

    finally:
        cur.close()
        release_connection(conn)

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
        return {
            "song_id": None,
            "title": None,
            "artist": None,
            "score": 0,
            "top_matches": [],
            "time_taken_s": (
                time.perf_counter()
                - start_time
            ),
        }

    conn = get_connection()
    cur = conn.cursor()

    rows = []
    cache_hits = 0
    cache_misses = 0

    try:
        batch_size = 200
        for start in range(
            0,
            len(query_hashes),
            batch_size,
        ):

            batch = query_hashes[
                start :
                start + batch_size
            ]

        keys = []
        lookup_data = []

        for (
            h,
            _,
            anchor_freq,
        ) in batch:

            h = int(h)
            anchor_freq = int(anchor_freq)

            keys.append(
                f"fp:{h}:{anchor_freq // 4}"
            )

            lookup_data.append(
                (
                    h,
                    anchor_freq,
                )
            )

        cached_results = redis_client.mget(
            keys
        )

        missing_batch = []

        for (
            cached,
            (
                h,
                anchor_freq,
            ),
        ) in zip(
            cached_results,
            lookup_data,
        ):

            if cached is not None:
                rows.extend(
                    pickle.loads(
                        cached
                    )
                )
                cache_hits += 1

            else:
                missing_batch.append(
                    (
                        h,
                        anchor_freq,
                    )
                )
                cache_misses += 1

            if missing_batch:
                conditions = []
                params = []

                for (
                    h,
                    anchor_freq,
                ) in missing_batch:

                    conditions.append(
                        """
                        (
                            hash = %s
                            AND anchor_freq
                            BETWEEN %s AND %s
                        )
                        """
                    )

                    params.extend(
                        [
                            h,
                            anchor_freq
                            - anchor_tol,
                            anchor_freq
                            + anchor_tol,
                        ]
                    )

                sql = f"""
                SELECT
                    hash,
                    song_id,
                    time_offset,
                    anchor_freq
                FROM fingerprints
                WHERE {" OR ".join(conditions)}
                """

                cur.execute(
                    sql,
                    params,
                )
                db_rows = (
                    cur.fetchall()
                )

                grouped = defaultdict(
                    list
                )
                for row in db_rows:

                    h = int(row[0])

                    grouped[h].append(
                        row
                    )

                    rows.append(
                        row
                    )

                for (
                    h,
                    anchor_freq,
                ) in missing_batch:

                    bucket = (
                        anchor_freq // 4
                    )

                    vals = grouped.get(h, [],
                    )
                    if len(vals) <= 100:
                        redis_client.setex(
                            f"fp:{h}:{bucket}",
                            3600,
                            pickle.dumps(vals),
                        )

        # print(
        #     f"Redis Hits: "
        #     f"{cache_hits} | "
        #     f"Redis Misses: "
        #     f"{cache_misses}"
        # )

        # print(
        #     f"Rows fetched: "
        #     f"{len(rows)}"
        # )

        db_map = defaultdict(
            list
        )

        for (
            h,
            song_id,
            db_offset,
            anchor_freq,
        ) in rows:

            db_map[h].append(
                (
                    song_id,
                    db_offset,
                )
            )

        delta_votes = defaultdict(
            list
        )

        for (
            h,
            q_offset,
            _anchor,
        ) in query_hashes:

            h = int(h)
            q_offset = int(
                q_offset
            )

            for (
                song_id,
                db_offset,
            ) in db_map.get(
                h,
                [],
            ):

                delta_votes[
                    song_id
                ].append(
                    db_offset
                    - q_offset
                )

        song_scores = []

        for (
            song_id,
            deltas,
        ) in delta_votes.items():

            score = Counter(
                deltas
            ).most_common(1)[0][1]

            if (
                score
                >= min_score
            ):

                song_scores.append(
                    (
                        song_id,
                        score,
                    )
                )

        song_scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        top_song_ids = [
            song_id
            for song_id, _
            in song_scores[:5]
        ]

        song_info = {}

        if top_song_ids:
            cur.execute(
                """
                SELECT
                    id,
                    title,
                    artist
                FROM songs
                WHERE id = ANY(%s)
                """,
                (
                    top_song_ids,
                ),
            )
            for (
                sid,
                title,
                artist,
            ) in cur.fetchall():

                song_info[
                    int(sid)
                ] = {
                    "title": title,
                    "artist": artist,
                }
    finally:
        cur.close()
        release_connection(
            conn
        )

    top_matches = []

    for (
        song_id,
        score,
    ) in song_scores[:5]:

        info = song_info.get(
            song_id,
            {},
        )

        top_matches.append(
            {
                "song_id": int(
                    song_id
                ),
                "title": info.get(
                    "title"
                ),
                "artist": info.get(
                    "artist"
                ),
                "score": int(
                    score
                ),
            }
        )

    best_match = (
        top_matches[0]
        if top_matches
        else None
    )

    return {
        "song_id": (
            best_match["song_id"]
            if best_match
            else None
        ),
        "title": (
            best_match["title"]
            if best_match
            else None
        ),
        "artist": (
            best_match["artist"]
            if best_match
            else None
        ),
        "score": (
            best_match["score"]
            if best_match
            else 0
        ),
        "top_matches": top_matches,
        "time_taken_s": (
            time.perf_counter()
            - start_time
        ),
    }

def search_song_by_audio_path(
    audio_path,
    *,
    anchor_tol=2,
    min_score=10,
    hash_kwargs=None,
):
    hash_kwargs = hash_kwargs or {}
    query_hashes, _ = generate_hashes_for_audio(
        audio_path,
        **hash_kwargs,
    )
    return match_audio_hashes(
        query_hashes,
        anchor_tol=anchor_tol,
        min_score=min_score,
    )