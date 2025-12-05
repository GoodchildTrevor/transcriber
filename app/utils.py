import gc
from logging import Logger
import os
from typing import Any, Optional
import tempfile
import shutil
import subprocess

from fastapi import FastAPI, UploadFile
import librosa
import numpy as np
import torch
import soundfile as sf
import whisperx

from app.config import BATCH_SIZE


def load_audio_from_uploadfile(file: UploadFile, target_sr: int = 16000) -> np.ndarray:
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".tmp", mode="wb") as tmp:
            file.file.seek(0)
            shutil.copyfileobj(file.file, tmp)
            tmp.flush()
            
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads", "0",
                "-i", tmp.name,
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(target_sr),
                "-"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {err.decode()}")

            audio_np = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            
            return audio_np

    except Exception as e:
        raise RuntimeError(f"Audio decoding failed: {e}") from e


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
                
                # Конвертируем audio в torch tensor
                audio_tensor = torch.from_numpy(audio).float()
                # Добавляем размер канала: [channels, samples] -> [1, samples] для моно
                audio_tensor = audio_tensor.unsqueeze(0)
                
                logger.info(f"Audio tensor shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")
                
                if num_participants:
                    diarize_segments = diarize_model(
                        {
                            "waveform": audio_tensor,
                            "sample_rate": 16000
                        },
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                else:
                    diarize_segments = diarize_model(
                        {
                            "waveform": audio_tensor,
                            "sample_rate": 16000
                        }
                    )
                
                logger.info("Segments are diarized")
                aligned_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                logger.info("Speakers assigned successfully")
            
                finally:
                    # Cleanup
                    if audio is not None:
                        del audio
                    if result is not None:
                        del result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
