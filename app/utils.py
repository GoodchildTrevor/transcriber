import gc
from logging import Logger
import os
from typing import Any, Optional
import tempfile
import shutil

from fastapi import FastAPI, UploadFile
import librosa
import numpy as np
import torch
import soundfile as sf
import whisperx

from app.config import BATCH_SIZE


def load_audio_from_uploadfile(file: UploadFile, target_sr: int = 16000) -> np.ndarray:
    """
    Load and preprocess audio directly from UploadFile (streaming, low memory).
    
    Uses temporary file; ensures cleanup and resets file pointer.
    """
    try:
        file.file.seek(0)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".tmp", mode="wb") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())

            audio, sr = sf.read(tmp.name, dtype='float32', always_2d=True)

        try:
            file.file.seek(0)
        except (OSError, AttributeError):
            pass 

        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0]

        if sr != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=target_sr,
                res_type='kaiser_best'
            )
        return audio

    except Exception as e:
        raise RuntimeError(f"Audio decoding/resampling failed: {e}") from e


async def transcriber(
        app: FastAPI,
        logger: Logger,
        upload_file: UploadFile,
        language: str = "auto",
        num_participants: Optional[int] = None,
        diarization: Optional[bool] = None
    ) -> list[dict[str, Any]]:
    """
    Transcribe and optionally diarize an audio file.

    :param logger: Logger instance
    :param upload_file: Uploaded audio file
    :param num_participants: Expected number of speakers (for diarization)
    :param diarization: Whether to run speaker diarization
    :return: List of aligned segments
    """

    audio = None
    result = None

    try:
        # 1️⃣ Load audio as np.ndarray
        upload_file.file.seek(0)
        audio = load_audio_from_uploadfile(upload_file, target_sr=16000)

        min_speakers = 1
        max_speakers = num_participants

        logger.info("Loading Whisper model")
        whisper_model = app.state.whisper_model
        device = app.state.device  

        force_lang = None
        if language != "auto":
            force_lang = language
            logger.info(f"Language forced to: '{language}'")
        else:
            logger.info("Language detection enabled")

        with torch.no_grad():
            # 2️⃣ Transcription
            try:
                result = whisper_model.transcribe(
                    audio,
                    language=force_lang,
                    batch_size=BATCH_SIZE,
                )
                detected_lang = result.get("language", "unknown")
            except Exception as e:
                raise RuntimeError(f"Whisper transcription failed: {e}") from e

            # 3️⃣ Alignment
            if detected_lang == "ru" and hasattr(app.state, "align_ru_model"):
                align_model = app.state.align_ru_model
                align_meta = app.state.align_ru_metadata
                logger.debug("Using preloaded Russian align model")
            else:
                align_model, align_meta = app.state.load_align_model_cached(detected_lang)
            try:
                segments = result["segments"]  # список сегментов
                aligned_result = whisperx.align(
                    segments,
                    align_model,
                    align_meta,
                    audio,
                    device,
                    return_char_alignments=False
                )
            except Exception as e:
                raise RuntimeError(f"Alignment failed: {e}") from e

            logger.info("Segments are aligned successfully")

            # 4️⃣ Optional diarization
            if diarization:
                diarize_model = app.state.diarize_model
                logger.info(f"Diarization model loaded, max_speakers={max_speakers}")
                if num_participants:
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                else:
                    diarize_segments = diarize_model(audio)

                logger.info("Segments are diarized")
                aligned_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                logger.info("Speakers assigned successfully")

        return aligned_result["segments"]

    finally:
        # Cleanup
        if audio is not None:
            del audio
        if result is not None:
            del result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
