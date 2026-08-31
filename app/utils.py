import asyncio
from dotenv import load_dotenv
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


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")



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
    diarization: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """
    Transcribe and optionally diarize an audio file.


    :param app: FastAPI instance with preloaded models in app.state
    :param logger: Logger instance
    :param upload_file: Uploaded audio file
    :param language: 'auto' or language code (e.g. 'ru')
    :param num_participants: Expected number of speakers (for diarization)
    :param diarization: Whether to run speaker diarization
    :return: List of aligned segments
    """


    audio: Optional[np.ndarray] = None
    result: Optional[dict[str, Any]] = None
    aligned_result: Optional[dict[str, Any]] = None


    try:
        # 1⃣⃣ Load audio as np.ndarray (синхронный код в thread pool можно не выносить)
        upload_file.file.seek(0)
        audio = load_audio_from_uploadfile(upload_file, target_sr=16000)


        min_speakers = 1
        max_speakers = num_participants


        device = app.state.device
        whisper_model = app.state.whisper_model


        # Empty query/form values must not enable language detection;
        # Russian is the service default. Pass "auto" explicitly to detect.
        language = (language or "ru").strip().lower()
        force_lang = None if language == "auto" else language

        if force_lang:
            logger.info("Language forced to: %r", force_lang)
        else:
            logger.info("Language detection enabled")


        loop = asyncio.get_event_loop()


        # 2⃣ Transcription with CUDA→CPU fallback
        with torch.no_grad():
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: whisper_model.transcribe(
                        audio,
                        language=force_lang,
                        batch_size=BATCH_SIZE,
                    ),
                )
            except RuntimeError as e:
                msg = str(e)
                if "CUDA out of memory" in msg or "CUDA failed with error out of memory" in msg:
                    logger.warning("CUDA OOM during ASR, retrying on CPU")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    # ленивый CPU-модель-кеш в app.state
                    if not hasattr(app.state, "whisper_model_cpu"):
                        app.state.whisper_model_cpu = whisperx.load_model(
                            MODEL_NAME,
                            device="cpu",
                            compute_type="int8",
                        )
                        logger.info("Loaded CPU Whisper model (int8) for fallback")


                    whisper_cpu = app.state.whisper_model_cpu
                    device = "cpu"


                    result = await loop.run_in_executor(
                        None,
                        lambda: whisper_cpu.transcribe(
                            audio,
                            language=force_lang,
                            batch_size=1,
                        ),
                    )
                else:
                    raise RuntimeError(f"Whisper transcription failed: {e}") from e


        detected_lang = result.get("language")
        if not detected_lang:
            raise RuntimeError("Whisper could not detect the language")


        # 3⃣ Alignment (с reuse ru-модели + кэш для остальных языков)
        if detected_lang == "ru" and hasattr(app.state, "align_ru_model"):
            align_model = app.state.align_ru_model
            align_meta = app.state.align_ru_metadata
            logger.debug("Using preloaded Russian align model")
        else:
            align_model, align_meta = app.state.load_align_model_cached(detected_lang)


        try:
            segments = result["segments"]
            aligned_result = whisperx.align(
                segments,
                align_model,
                align_meta,
                audio,
                device,
                return_char_alignments=False,
            )
        except Exception as e:
            raise RuntimeError(f"Alignment failed: {e}") from e


        logger.info("Segments are aligned successfully")


        # 4⃣ Optional diarization with CUDA→CPU fallback
        if diarization:
            diarize_model = app.state.diarize_model
            logger.info(f"Diarization model loaded, max_speakers={max_speakers}")


            try:
                diarize_segments = await loop.run_in_executor(
                    None,
                    lambda: diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    if num_participants
                    else diarize_model(audio),
                )
            except RuntimeError as e:
                msg = str(e)
                if "CUDA out of memory" in msg or "CUDA failed with error out of memory" in msg:
                    logger.warning("CUDA OOM during diarization, retrying on CPU")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    diarize_model_cpu = DiarizationPipeline(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=HF_TOKEN,
                        device="cpu",
                    )


                    diarize_segments = await loop.run_in_executor(
                        None,
                        lambda: diarize_model_cpu(
                            audio,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                        )
                        if num_participants
                        else diarize_model_cpu(audio),
                    )
                else:
                    raise RuntimeError(f"Diarization failed: {e}") from e


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
        if aligned_result is not None:
            del aligned_result


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
