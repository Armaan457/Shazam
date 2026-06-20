import sqlite3
import tempfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import librosa
import os
import soundfile as sf

from utils import insert_song_fingerprints, fingerprint_worker

dataset_dir = Path(__file__).resolve().parents[1] / "data" / "dataset" / "genres_original"
db_path = Path(__file__).resolve().with_name("music_database.db")

def list_audio_files(dataset_dir: Path) -> list[Path]:
    audio_files: list[Path] = []
    for genre_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for audio_path in sorted(genre_dir.iterdir()):
            if audio_path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                audio_files.append(audio_path)
    return audio_files


def make_query_clip(audio_path: Path, *, clip_start_s: float, clip_duration_s: float, sr: int) -> Path:
    y, _ = librosa.load(
        audio_path,
        sr=sr,
        offset=clip_start_s,
        duration=clip_duration_s,
    )

    if y.size == 0:
        raise ValueError(f"Could not extract a query clip from {audio_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="shazam_query_"))
    clip_path = tmp_dir / f"{audio_path.stem}_query.wav"
    sf.write(clip_path, y, sr)
    return clip_path


def build_database_batch(
    audio_files,
    *,
    clean_db=False,
    commit_every=100,
):
    if clean_db and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA cache_size=-100000"
    )

    conn.execute(
        "PRAGMA temp_store=MEMORY"
    )

    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            artist TEXT,
            duration FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash BIGINT NOT NULL,
            song_id INT NOT NULL,
            time_offset INT NOT NULL,
            anchor_freq INT NOT NULL,
            FOREIGN KEY (song_id)
            REFERENCES songs(id)
            ON DELETE CASCADE
        )
        """
    )

    cur.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_hash_anchor
    ON fingerprints(hash, anchor_freq)
    """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hash_anchor_song
        ON fingerprints(
            hash,
            anchor_freq,
            song_id
        )
        """
    )
    conn.commit()
    cur.close()
    workers = max(
        1,
        os.cpu_count() - 1,
    )
    successful = 0
    failed = 0
    corrupt_files = 0

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        results = executor.map(
            fingerprint_worker,
            audio_files,
        )

        for idx, item in enumerate(
            tqdm(
                results,
                total=len(audio_files),
                desc="Indexing Songs",
                unit="song",
            ),
            start=1,
        ):  
            if not item["success"]:
                corrupt_files += 1
                tqdm.write(
                    f"SKIPPED: "
                    f"{Path(item['path']).name} "
                    f"({item['error']})"
                )
                continue

            try:

                insert_song_fingerprints(
                    conn,
                    title=item["title"],
                    artist=item["artist"],
                    duration=item["duration"],
                    hashes=item["hashes"],
                )

                successful += 1

                if idx % commit_every == 0:
                    conn.commit()
            except Exception as e:
                failed += 1
                tqdm.write(
                    f"FAILED {item['title']}: "
                    f"{type(e).__name__}: {e}"
                )
    conn.commit()
    conn.close()
    print("\nDatabase Build Complete")
    print(
        f"Indexed: {successful}"
    )
    print(
        f"Failed: {failed}"
    )
    print(
        f"Corrupt Files: {corrupt_files}"
    )