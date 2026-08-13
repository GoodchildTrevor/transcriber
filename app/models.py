import logging
from typing import Optional

from fastapi import HTTPException, UploadFile
import mimetypes
from pydantic import BaseModel, Field

from app.config import ALLOWED_AUDIO_TYPES, ALLOWED_VIDEO_TYPES, MAX_FILE_SIZE

logger = logging.getLogger(__name__)


class ValidatedAudioFile:
    """
    Dependency class for validating uploaded audio OR video files.
    Audio will be extracted from video using ffmpeg.
    :param max_size: Maximum allowed file size in bytes
    :param allowed_media_types: Set of allowed MIME types (audio + video)
    """
    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE,
        allowed_media_types: set = None,
    ):
        self.max_size = max_size
        self.allowed_media_types = allowed_media_types or (ALLOWED_AUDIO_TYPES | ALLOWED_VIDEO_TYPES)
        self.allowed_audio_types = ALLOWED_AUDIO_TYPES
        self.allowed_video_types = ALLOWED_VIDEO_TYPES

    def _guess_mime_type(self, filename: str) -> Optional[str]:
        """Guess MIME type from filename extension"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    async def __call__(self, file: UploadFile) -> UploadFile:
        """
        Validate uploaded file (audio or video).
        :param file: File uploaded by user
        :return: Validated UploadFile
        :raises HTTPException: On validation failure
        """
        content_type = file.content_type

        if content_type == "application/octet-stream" or content_type not in self.allowed_media_types:
            guessed_type = self._guess_mime_type(file.filename)
            if guessed_type and guessed_type in self.allowed_media_types:
                content_type = guessed_type
                logger.info(f"Corrected MIME type from {file.content_type} to {content_type} for {file.filename}")

        logger.info(f"Validating file: {file.filename}, type: {content_type}, size: {file.size}")

        if content_type not in self.allowed_media_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {file.content_type}. Allowed: {self.allowed_media_types}"
            )

        if file.size and file.size > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file.size} bytes > {self.max_size} bytes"
            )

        if content_type in self.allowed_video_types:
            logger.info(f"Video file detected: {file.filename} (type: {content_type}). Audio will be extracted.")

        return file


class TranscriptionParams(BaseModel):
    num_participants: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Ожидаемое число спикеров. None -> диаризация определит число спикеров автоматически.",
    )
    diarization: bool = Field(
        default=False,
        description="Enable speaker diarization",
    )
    language: str = Field(
        default="auto",
        description="Language of audio",
    )
