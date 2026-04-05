import os
import tempfile
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from app.models import SongSearchOutput, SongUploadInput, SongUploadOutput
from app.utils import generate_hashes_for_audio, ingest_song_from_audio, match_audio_hashes

router = APIRouter()


def _parse_song_upload_input(
    title: Optional[str] = Form(default=None),
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
        query_hashes = generate_hashes_for_audio(temp_path)
        result = match_audio_hashes(query_hashes)
        print(f"Search completed in {result['time_taken_s']} s")
        return SongSearchOutput.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to search song: {exc}") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
