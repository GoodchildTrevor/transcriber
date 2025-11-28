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
        num_participants: Optional[int] = None,
        diarization: Optional[bool] = None
    ) -> list[dict[str, Any]]:
    """
    Transcribe and diarize audio.

    :param logger: Logger instance
    :param audio_bytes: audio file content (e.g., MP3, WAV)
    :param num_participants: Expected number of speakers (for diarization)
    :return: Formatted transcript with speaker labels, e.g. "[SPEAKER_00]: Hello"
    """

    audio = None
    result = None

    try:
        upload_file.seek(0)
        audio = load_audio_from_uploadfile(upload_file, target_sr=16000)

        min_speakers = 1
        max_speakers = num_participants

        logger.info("Loading model")
        model = app.state.whisper_model

        with torch.no_grad():
            try:
                result = model.transcribe(
                    audio, 
                    batch_size=BATCH_SIZE,
                )
            except Exception as e:
                raise RuntimeError(f"Whisper transcription failed: {e}") from e

            model_a = app.state.align_model
            metadata =app.state.align_metadata
            try:
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    return_char_alignments=False
                )
            except Exception as e:
                raise RuntimeError(f"Alignment failed: {e}") from e
        
            logger.info("Segments are aligned! Result recieved!")

            if diarization:
                diarize_model = app.state.diarize_model
                logger.info(f"Diarization model loaded, max_speakers= {max_speakers}")
                if num_participants:
                    diarize_segments = diarize_model(
                        audio, 
                        min_speakers=min_speakers, 
                        max_speakers=max_speakers
                    )
                else:
                    diarize_segments = diarize_model(audio)
                logger.info("Segments are diarized")
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("Speakers are assigned!")

        segments = result["segments"]

        return segments
    
    finally:
        if audio is not None:
            del audio
        if result is not None:
            del result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
