from __future__ import annotations

import shutil
import time
from pathlib import Path
import numpy as np

import utils as shazam_utils
from setup import build_database_batch, list_audio_files, make_query_clip
from utils import search_song_by_audio_path


dataset_dir = Path(__file__).resolve().parents[1] / "data" / "fma_small" 
db_path = Path(__file__).resolve().with_name("music_database.db")
DEFAULT_SR = 22050
DEFAULT_QUERY_START_S = 10.0
DEFAULT_QUERY_DURATION_S = 5.0
DEFAULT_ANCHOR_TOL = 2
DEFAULT_MIN_SCORE = 10


def evaluate(
    audio_files: list[Path],
    *,
    sr: int,
    query_start_s: float,
    query_duration_s: float,
    anchor_tol: int,
    min_score: int,
) -> dict[str, object]:

    total = 0

    top1_correct = 0
    top5_correct = 0
    matched = 0

    prediction_times: list[float] = []

    for index, audio_path in enumerate(audio_files, start=1):
        try:
            query_clip_path = make_query_clip(
                audio_path,
                clip_start_s=query_start_s,
                clip_duration_s=query_duration_s,
                sr=sr,
            )
        except Exception as e:
            print(
                f"[{index}/{len(audio_files)}] {audio_path.name}: "
                f"SKIPPED - {type(e).__name__}: {e}"
            )
            continue

        try:
            result = search_song_by_audio_path(
            str(query_clip_path),
            anchor_tol=anchor_tol,
            min_score=min_score,
            )
            prediction_times.append(
                result["time_taken_s"]
            )

        except Exception as e:
            print(
                f"[{index}/{len(audio_files)}] {audio_path.name}: "
                f"SKIPPED - {type(e).__name__}: {e}"
            )

            shutil.rmtree(
                query_clip_path.parent,
                ignore_errors=True,
            )

            continue

        finally:
            shutil.rmtree(
                query_clip_path.parent,
                ignore_errors=True,
            )

        total += 1

        top_matches = result.get(
            "top_matches",
            [],
        )

        if top_matches:
            matched += 1

        gt_title = audio_path.stem

        predicted_title = (
            top_matches[0]["title"]
            if top_matches
            else None
        )

        top5_titles = [
            match["title"]
            for match in top_matches
        ]

        is_top1 = (
            predicted_title == gt_title
        )

        is_top5 = (
            gt_title in top5_titles
        )

        if is_top1:
            top1_correct += 1

        if is_top5:
            top5_correct += 1

        print(
            f"[{index}/{len(audio_files)}] "
            f"{audio_path.name}: "
            f"pred={predicted_title!r} "
            f"score={result['score']} "
            f"top1={'yes' if is_top1 else 'no'} "
            f"top5={'yes' if is_top5 else 'no'} "
        )

    top1_accuracy = (
        top1_correct / total
        if total
        else 0.0
    )

    top5_accuracy = (
        top5_correct / total
        if total
        else 0.0
    )

    coverage = (
        matched / total
        if total
        else 0.0
    )

    return {
        "total": total,
        "matched": matched,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "coverage": coverage,
        "prediction_times": prediction_times,
    }

def main():
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    audio_files = list_audio_files(dataset_dir)

    if not audio_files:
        raise RuntimeError(
            f"No audio files found under {dataset_dir}"
        )

    shazam_utils.DB_PATH = str(db_path)
    print(
        f"Found {len(audio_files)} audio files"
    )

    REBUILD_DATABASE = False
    if REBUILD_DATABASE or not db_path.exists():
        print(
            "Building fingerprint database..."
        )
        build_database_batch(
            audio_files,
            clean_db=True,
        )
    else:
        print(
            f"Using existing database: "
            f"{db_path}"
        )

    results = evaluate(
        audio_files,
        sr=DEFAULT_SR,
        query_start_s=DEFAULT_QUERY_START_S,
        query_duration_s=DEFAULT_QUERY_DURATION_S,
        anchor_tol=DEFAULT_ANCHOR_TOL,
        min_score=DEFAULT_MIN_SCORE,
    )

    print(
        f"Database Size: "
        f"{results['total']} songs"
    )
    print(
        f"Matched Predictions: "
        f"{results['matched']}"
    )
    print(
        f"Top-1 Accuracy: "
        f"{results['top1_accuracy']:.4%}"
    )
    print(
        f"Top-5 Accuracy: "
        f"{results['top5_accuracy']:.4%}"
    )
    print(
        f"Coverage: "
        f"{results['coverage']:.4%}"
    )
    if results["prediction_times"]:
        latencies = results["prediction_times"]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(
            latencies,
            95,
        )
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        print("\nLatency Statistics")
        print(
            f"Average Latency: "
            f"{avg_latency:.4f}s "
            f"({avg_latency * 1000:.2f} ms)"
        )
        print(
            f"P95 Latency: "
            f"{p95_latency:.4f}s "
            f"({p95_latency * 1000:.2f} ms)"
        )
        print(
            f"Minimum Latency: "
            f"{min_latency:.4f}s"
        )
        print(
            f"Maximum Latency: "
            f"{max_latency:.4f}s"
        )

if __name__ == "__main__":
    main()