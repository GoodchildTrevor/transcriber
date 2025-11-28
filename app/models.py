from fastapi import HTTPException, UploadFile
import logging
from typing import Optional
from pydantic import BaseModel, Field

from app.config import ALLOWED_AUDIO_TYPES, MAX_FILE_SIZE

logger = logging.getLogger(__name__)


class ValidatedAudioFile:
    """
    Dependency class for validating uploaded audio files.

    :param max_size: Maximum allowed file size in bytes
    :param allowed_types: Set of allowed MIME types
    """

    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE,
        allowed_types: set = ALLOWED_AUDIO_TYPES
    ):
        self.max_size = max_size
        self.allowed_types = allowed_types

    async def __call__(self, file: UploadFile) -> UploadFile:
        """
        Validate uploaded file.

        :param file: File uploaded by user
        :return: Validated UploadFile
        :raises HTTPException: On validation failure
        """
        logger.info(f"Validating file: {file.filename}, type: {file.content_type}, size: {file.size}")

        if file.content_type not in self.allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {file.content_type}. Allowed: {self.allowed_types}"
            )

        if file.size > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file.size} bytes > {self.max_size} bytes"
            )

        return file


class TranscriptionParams(BaseModel):
    num_participants: Optional[int] = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of speakers",
    )
    diarization: bool = Field(
        default=False,
        description="Enable speaker diarization",
    )
