from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path
import librosa
import numpy as np

import utils as shazam_utils
from setup import list_audio_files
from noise import create_noisy_query
from utils import search_song_by_audio_path

DEFAULT_SR = 22050
DEFAULT_QUERY_DURATION_S = 5.0
DEFAULT_ANCHOR_TOL = 2
DEFAULT_MIN_SCORE = 10
ROBUSTNESS_SAMPLE_SIZE = 2000
N_QUERIES_PER_SONG = 3

DISTORTIONS = [
    {
        "name": "clean",
    },
    {
        "name": "noise_002",
        "noise_factor": 0.02,
    },
    {
        "name": "noise_005",
        "noise_factor": 0.05,
    },
    {
        "name": "pitch_up",
        "pitch_shift_steps": 1,
    },
    {
        "name": "pitch_down",
        "pitch_shift_steps": -1,
    },
    {
        "name": "stretch_fast",
        "time_stretch_rate": 1.05,
    },
    {
        "name": "stretch_slow",
        "time_stretch_rate": 0.95,
    },
    {
        "name": "phone",
        "use_phone_simulation": True,
    },
    {
        "name": "reverb",
        "use_reverb": True,
    },
    {
        "name": "realistic",
        "noise_factor": 0.02,
        "use_phone_simulation": True,
        "use_compression": True,
    },
]

DISTORTION_WEIGHTS = [10] * len(DISTORTIONS)

def evaluate_robustness(
    audio_files,
    *,
    sr,
    query_duration_s,
    anchor_tol,
    min_score,
):

    total_queries = 0
    total_top1 = 0
    total_top5 = 0
    total_matched = 0

    latencies = []

    stats = defaultdict(
        lambda: {
            "total": 0,
            "top1": 0,
            "top5": 0,
            "matched": 0,
            "scores": [],
            "latencies": [],
        }
    )

    for song_idx, audio_path in enumerate(audio_files, start=1):
        try:
            duration = librosa.get_duration(
                path=str(audio_path)
            )
        except Exception as e:
            print(
                f"Skipping {audio_path.name}: {e}"
            )
            continue

        if duration <= query_duration_s:
            continue

        for _ in range(N_QUERIES_PER_SONG):
            distortion = random.choices(
                DISTORTIONS,
                weights=DISTORTION_WEIGHTS,
                k=1,
            )[0]
            distortion_name = distortion["name"]

            start_time = random.uniform(
                0,
                duration - query_duration_s,
            )
            try:

                query_clip_path = create_noisy_query(
                    audio_path=audio_path,
                    sr=sr,
                    clip_start_s=start_time,
                    clip_duration_s=query_duration_s,

                    noise_factor=distortion.get(
                        "noise_factor",
                        0.0,
                    ),
                    pitch_shift_steps=distortion.get(
                        "pitch_shift_steps",
                        0.0,
                    ),
                    time_stretch_rate=distortion.get(
                        "time_stretch_rate",
                        1.0,
                    ),
                    use_reverb=distortion.get(
                        "use_reverb",
                        False,
                    ),
                    use_phone_simulation=distortion.get(
                        "use_phone_simulation",
                        False,
                    ),
                    use_compression=distortion.get(
                        "use_compression",
                        False,
                    ),
                )

            except Exception as e:
                print(
                    f"Query generation failed "
                    f"for {audio_path.name}: {e}"
                )
                continue

            try:
                result = search_song_by_audio_path(
                    str(query_clip_path),
                    anchor_tol=anchor_tol,
                    min_score=min_score,
                )

            except Exception as e:
                print(
                    f"Search failed "
                    f"for {audio_path.name}: {e}"
                )
                continue

            finally:
                shutil.rmtree(
                    query_clip_path.parent,
                    ignore_errors=True,
                )

            total_queries += 1

            score = result.get(
                "score",
                0,
            )

            latency = result.get(
                "time_taken_s",
                0,
            )

            latencies.append(latency)

            gt_title = audio_path.stem

            top_matches = result.get(
                "top_matches",
                [],
            )

            predicted_title = (
                top_matches[0]["title"]
                if top_matches
                else None
            )

            top5_titles = [
                m["title"]
                for m in top_matches
            ]

            is_top1 = (
                predicted_title == gt_title
            )

            is_top5 = (
                gt_title in top5_titles
            )

            is_matched = bool(top_matches)

            if is_top1:
                total_top1 += 1

            if is_top5:
                total_top5 += 1

            if is_matched:
                total_matched += 1

            d = stats[distortion_name]

            d["total"] += 1

            if is_top1:
                d["top1"] += 1

            if is_top5:
                d["top5"] += 1

            if is_matched:
                d["matched"] += 1

            d["scores"].append(score)
            d["latencies"].append(latency)

            print(
            f"[{song_idx}/{len(audio_files)}] "
            f"{distortion_name:<15} "
            f"score={score} "
            f"top1={'yes' if is_top1 else 'no'} "
            f"top5={'yes' if is_top5 else 'no'}"
            )

    return {
        "total_queries": total_queries,

        "top1_accuracy": (
            total_top1 / total_queries
            if total_queries else 0
        ),

        "top5_accuracy": (
            total_top5 / total_queries
            if total_queries else 0
        ),

        "coverage": (
            total_matched / total_queries
            if total_queries else 0
        ),

        "avg_latency": (
            float(np.mean(latencies))
            if latencies else 0
        ),

        "p95_latency": (
            float(np.percentile(latencies, 95))
            if latencies else 0
        ),

        "distortion_stats": stats,
    }

if __name__ == "__main__":

    dataset_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "fma_small"
    )

    db_path = Path(__file__).resolve().with_name(
        "music_database.db"
    )

    shazam_utils.DB_PATH = str(db_path)

    audio_files = list_audio_files(
        dataset_dir
    )

    audio_files = random.sample(
        audio_files,
        min(
            ROBUSTNESS_SAMPLE_SIZE,
            len(audio_files),
        ),
    )

    print(
        f"Evaluating {len(audio_files)} songs "
        f"with {N_QUERIES_PER_SONG} queries/song"
    )

    results = evaluate_robustness(
        audio_files,
        sr=DEFAULT_SR,
        query_duration_s=DEFAULT_QUERY_DURATION_S,
        anchor_tol=DEFAULT_ANCHOR_TOL,
        min_score=DEFAULT_MIN_SCORE,
    )

    print(
        f"Queries: "
        f"{results['total_queries']}"
    )
    print(
        f"Top1 Accuracy: "
        f"{results['top1_accuracy']:.2%}"
    )
    print(
        f"Top5 Accuracy: "
        f"{results['top5_accuracy']:.2%}"
    )
    print(
        f"Coverage: "
        f"{results['coverage']:.2%}"
    )
    print(
        f"Average Latency: "
        f"{results['avg_latency'] * 1000:.2f} ms"
    )
    print(
        f"P95 Latency: "
        f"{results['p95_latency'] * 1000:.2f} ms"
    )

    for name, d in sorted(
        results["distortion_stats"].items()
    ):
        if d["total"] == 0:
            continue
        print(
            f"{name:<15}"
            f"N={d['total']:<5} "
            f"Top1={(d['top1']/d['total']):.2%} "
            f"Top5={(d['top5']/d['total']):.2%} "
            f"Coverage={(d['matched']/d['total']):.2%} "
            f"AvgScore={np.mean(d['scores']):.1f}"
        )