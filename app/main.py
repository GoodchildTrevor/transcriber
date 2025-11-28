import asyncio
import os
import logging

from fastapi import FastAPI, HTTPException, Query, UploadFile, Depends
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import torch
import whisperx

from app.models import TranscriptionParams, ValidatedAudioFile
from app.utils import transcriber
from app.config import (
    COMPUTE_TYPE,
    LANGUAGE,
    MODEL_NAME,
    WHISPERX_THREADS,
    MAX_CONCURRENT_TRANSCRIPTIONS
)

load_dotenv()

os.environ["TORCH_WEIGHTS_ONLY_LOAD_ALLOW_UNSAFE"] = "True"
HF_TOKEN = os.getenv("HF_TOKEN")

LOG_PATH = "transcriber.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
transcription_semaphor = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)


@app.on_event("startup")
async def startup_event():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    app.state.whisper_model = whisperx.load_model(
            MODEL_NAME,
            device=device,
            compute_type=COMPUTE_TYPE,
            language=LANGUAGE,
            # threads = WHISPERX_THREADS,
        )
    app.state.align_model, app.state.align_metadata = whisperx.load_align_model(
            language_code=LANGUAGE, 
            device=device
        )
    app.state.diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    ).to(device)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Start shutdown")
    if hasattr(app.state, "whisper_model"):
        del app.state.whisper_model
    if hasattr(app.state, "align_model"):
        del app.state.align_model
    if hasattr(app.state, "diarize_model"):
        del app.state.diarize_model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Shutdown has been completed")


@app.post("/transcriber/")
async def upload_file(
    file: UploadFile = Depends(ValidatedAudioFile()),
    num_participants: int = Query(1, ge=1, le=100),
    diarization: bool = Query(False)
    ):

    params = TranscriptionParams(
        num_participants=num_participants,
        diarization=diarization,
    )

    async with transcription_semaphor:
        logger.info(f"Processing: {file.filename}, {params.model_dump()}")

        try: 
            result = await transcriber(
                app=app, 
                logger=logger, 
                upload_file=file, 
                num_participants=num_participants, 
                diarization=diarization,
            )

            return JSONResponse(
                    content={"segments": result},
                )
        except RuntimeError as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("Unexpected error in transcription")
            raise HTTPException(status_code=500, detail="Internal transcription error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
