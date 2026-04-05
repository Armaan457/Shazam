from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SongUploadInput(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    artist: Optional[str] = Field(default=None, min_length=1, max_length=255)

    @field_validator("title", "artist", mode="before")
    @classmethod
    def normalize_text(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class SongUploadOutput(BaseModel):
    song_id: int
    title: str = Field(min_length=1, max_length=255)
    artist: Optional[str] = Field(default=None, min_length=1, max_length=255)
    duration: float = Field(gt=0)
    hash_count: int = Field(ge=0)


class SongSearchOutput(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    artist: Optional[str] = Field(default=None, min_length=1, max_length=255)
    time_taken_s: float = Field(ge=0)
