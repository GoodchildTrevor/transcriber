MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024

MAX_CONCURRENT_TRANSCRIPTIONS = 3

BATCH_SIZE = 16
COMPUTE_TYPE = "float32"
LANGUAGE = "ru"
MODEL_NAME = "large-v3-turbo"
WHISPERX_THREADS = 12

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac"
}
