from __future__ import annotations

import shutil
import time
from collections import defaultdict
from pathlib import Path

import utils as shazam_utils
from setup import build_database, list_audio_files, make_query_clip
from utils import search_song_by_audio_path


dataset_dir = Path(__file__).resolve().parents[1] / "data" / "dataset" / "genres_original"
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
    correct = 0
    matched = 0
    per_genre_total: dict[str, int] = defaultdict(int)
    per_genre_correct: dict[str, int] = defaultdict(int)
    prediction_times: list[float] = []

    for index, audio_path in enumerate(audio_files, start=1):
        genre = audio_path.parent.name
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
            start_time = time.time()
            result = search_song_by_audio_path(
                str(query_clip_path),
                anchor_tol=anchor_tol,
                min_score=min_score,
            )
            elapsed_time = time.time() - start_time
            prediction_times.append(elapsed_time)
        except Exception as e:
            print(
                f"[{index}/{len(audio_files)}] {audio_path.name}: "
                f"SKIPPED - {type(e).__name__}: {e}"
            )
            shutil.rmtree(query_clip_path.parent, ignore_errors=True)
            continue
        finally:
            shutil.rmtree(query_clip_path.parent, ignore_errors=True)

        total += 1
        per_genre_total[genre] += 1

        is_match = result["title"] == audio_path.stem
        if result["song_id"] is not None:
            matched += 1
        if is_match:
            correct += 1
            per_genre_correct[genre] += 1

        print(
            f"[{index}/{len(audio_files)}] {audio_path.name}: "
            f"pred={result['title']!r} score={result['score']} "
            f"match={'yes' if is_match else 'no'} time={elapsed_time:.4f}s"
        )

    overall_accuracy = correct / total if total else 0.0
    coverage = matched / total if total else 0.0
    per_genre_accuracy = {
        genre: (per_genre_correct[genre] / per_genre_total[genre])
        for genre in sorted(per_genre_total)
    }

    return {
        "total": total,
        "correct": correct,
        "matched": matched,
        "overall_accuracy": overall_accuracy,
        "coverage": coverage,
        "per_genre_accuracy": per_genre_accuracy,
        "prediction_times": prediction_times,
    }


def main() -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    audio_files = list_audio_files(dataset_dir)
    if not audio_files:
        raise RuntimeError(f"No audio files found under {dataset_dir}")

    shazam_utils.DB_PATH = str(db_path)

    print(f"Found {len(audio_files)} audio files")

    build_database(audio_files, clean_db=True)

    try:
        results = evaluate(
            audio_files,
            sr=DEFAULT_SR,
            query_start_s=DEFAULT_QUERY_START_S,
            query_duration_s=DEFAULT_QUERY_DURATION_S,
            anchor_tol=DEFAULT_ANCHOR_TOL,
            min_score=DEFAULT_MIN_SCORE,
        )
    finally:
        if db_path.exists():
            db_path.unlink()

    print("\nResults")
    print(f"Total tracks: {results['total']}")
    print(f"Correct matches: {results['correct']}")
    print(f"Matched predictions: {results['matched']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    
    if results['prediction_times']:
        avg_time = sum(results['prediction_times']) / len(results['prediction_times'])
        print(f"\nTiming Statistics")
        print(f"Average prediction time: {avg_time:.4f}s")
        print(f"Min prediction time: {min(results['prediction_times']):.4f}s")
        print(f"Max prediction time: {max(results['prediction_times']):.4f}s")


if __name__ == "__main__":
    main()