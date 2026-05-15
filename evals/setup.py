import sqlite3
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

import utils as shazam_utils
from utils import ingest_song_from_audio

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


def build_database(audio_files: list[Path], *, clean_db: bool) -> None:
    if clean_db and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
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
                FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON fingerprints(hash)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprint_hash_anchor ON fingerprints(hash, anchor_freq)"
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    for index, audio_path in enumerate(audio_files, start=1):
        try:
            ingest_song_from_audio(audio_path, title=audio_path.stem)
            print(f"[{index}/{len(audio_files)}] indexed {audio_path.name}")
        except Exception as e:
            print(f"[{index}/{len(audio_files)}] SKIPPED {audio_path.name}: {type(e).__name__}: {e}")