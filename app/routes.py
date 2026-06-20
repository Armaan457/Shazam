import os
import tempfile
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import ValidationError
from pathlib import Path
from app.models import SongSearchOutput, SongUploadInput, SongUploadOutput
from app.utils import ingest_song_from_audio, search_song_by_audio_path

router = APIRouter()

def _parse_song_upload_input(
    title: str = Form(...),
    artist: Optional[str] = Form(default=None),
) -> SongUploadInput:
    try:
        return SongUploadInput(title=title, artist=artist)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "upload.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        tmp.write(data)
        return tmp.name

static_dir = Path(__file__).resolve().parent / "static"

def mount_static_files(app):
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

@router.get("/", response_class=HTMLResponse)
def home():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Shazam</h1><p>Frontend is not available.</p>"


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/songs/upload", response_model=SongUploadOutput)
def upload_song(
    audio: UploadFile = File(...),
    song_input: SongUploadInput = Depends(_parse_song_upload_input),
):
    temp_path = _save_upload_to_temp(audio)
    try:
        result = ingest_song_from_audio(
            temp_path,
            title=song_input.title,
            artist=song_input.artist,
        )
        return SongUploadOutput.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to ingest song: {exc}") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/songs/search", response_model=SongSearchOutput)
def search_song(
    audio: UploadFile = File(...),
):
    temp_path = _save_upload_to_temp(audio)
    try:
        result = search_song_by_audio_path(
            temp_path,
            anchor_tol=2,
            min_score=10,
            hash_kwargs={
                "duration": 5,
            },
        )
        return SongSearchOutput(
            title=result["title"],
            artist=result["artist"],
            time_taken_s=result["time_taken_s"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to search song: {exc}") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
