import warnings

warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)

import logging
import os
import tempfile
import time
from typing import Any, Dict, Literal

import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pyannote.audio import Pipeline
from torchaudio.transforms import Resample

from model import ECAPA_gender

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey-patch torch.load to disable weights_only for PyTorch 2.6+ compatibility
# This is safe for trusted HF checkpoints
_original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

# Load model on startup
token = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "mhdp-africa/speaker-segmentation-callhome-voxconverse-diarization-v1")
GENDER_MODEL_ID = os.getenv("GENDER_MODEL_ID", "mhdp-africa/gender_classification_MHDP_asr_dataset_V1")
DEFAULT_INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "auto").lower()
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SECONDS = float(os.getenv("MIN_AUDIO_DURATION_SECONDS", "0.2"))

if DEFAULT_INFERENCE_DEVICE not in {"auto", "cpu", "gpu"}:
    DEFAULT_INFERENCE_DEVICE = "auto"

torch_cuda_available = torch.cuda.is_available()


def log_runtime_device_details() -> None:
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch_cuda_available}")
    if torch_cuda_available:
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        logger.info(
            "CUDA runtime details: "
            f"device_count={torch.cuda.device_count()}, "
            f"current_device={current_idx}, "
            f"device_name={props.name}, "
            f"total_memory_gb={round(props.total_memory / (1024 ** 3), 2)}"
        )
    else:
        logger.warning("Running without CUDA; inference will use CPU unless device mode is overridden.")


def resolve_device(device_mode: str) -> torch.device:
    mode = device_mode.lower()
    if mode == "gpu":
        if not torch_cuda_available:
            raise HTTPException(status_code=400, detail="GPU requested but CUDA is not available in this runtime.")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch_cuda_available else "cpu")


def runtime_device_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch_cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "default_inference_mode": DEFAULT_INFERENCE_DEVICE,
    }
    if torch_cuda_available:
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        snapshot["gpu_index"] = current_idx
        snapshot["gpu_name"] = props.name
        snapshot["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    else:
        snapshot["gpu_name"] = "not_available"
    return snapshot


_pipeline_cache: Dict[str, Pipeline] = {}
_gender_model_cache: Dict[str, ECAPA_gender] = {}


def get_pipeline_for_device(target_device: torch.device) -> Pipeline:
    cache_key = str(target_device)
    cached = _pipeline_cache.get(cache_key)
    if cached is not None:
        return cached

    logger.info(f"Loading model '{MODEL_ID}' on device: {target_device}")
    pipeline_instance = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline_instance.to(target_device)
    _pipeline_cache[cache_key] = pipeline_instance
    return pipeline_instance


def get_gender_model_for_device(target_device: torch.device) -> ECAPA_gender:
    cache_key = str(target_device)
    cached = _gender_model_cache.get(cache_key)
    if cached is not None:
        return cached

    logger.info(f"Loading gender model '{GENDER_MODEL_ID}' on device: {target_device}")
    model = ECAPA_gender.from_pretrained(GENDER_MODEL_ID)
    model.eval()
    model.to(target_device)
    _gender_model_cache[cache_key] = model
    return model


def load_audio_with_fallback(file_path: str) -> tuple:
    """
    Load audio file with fallback strategy.
    Tries torchaudio first, falls back to soundfile if torchaudio fails.
    Returns (waveform, sample_rate).
    """
    # Try torchaudio first
    torchaudio_error: Exception | None = None
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        logger.debug(f"Loaded audio via torchaudio: {file_path}")
        return waveform, sample_rate
    except Exception as e:
        torchaudio_error = e
        logger.warning(f"torchaudio.load failed ({type(e).__name__}), falling back to soundfile")

    # Fallback to soundfile
    try:
        data, sample_rate = sf.read(file_path, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T)
        logger.debug(f"Loaded audio via soundfile fallback: {file_path}")
        return waveform, sample_rate
    except Exception as e:
        logger.error(
            "Both decoders failed for file=%s, torchaudio_error=%s, soundfile_error=%s",
            file_path,
            repr(torchaudio_error),
            repr(e),
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Audio decode failed. "
                f"torchaudio={type(torchaudio_error).__name__ if torchaudio_error else 'None'}; "
                f"soundfile={type(e).__name__}"
            ),
        )


def resolve_audio_suffix(filename: str | None) -> str:
    suffix = os.path.splitext(filename or "audio.wav")[1].lower()
    return suffix if suffix else ".wav"


def validate_and_normalize_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    filename: str,
    stage: str,
) -> torch.Tensor:
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail=f"{stage}: invalid sample rate for file '{filename}'.")
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise HTTPException(status_code=400, detail=f"{stage}: empty or malformed audio for file '{filename}'.")
    if not torch.isfinite(waveform).all():
        raise HTTPException(status_code=400, detail=f"{stage}: audio has NaN/Inf values for file '{filename}'.")

    # Force mono normalization for consistent model input behavior.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.float()
    waveform = resample_waveform(waveform, sample_rate, target_sample_rate=TARGET_SAMPLE_RATE)

    duration_seconds = waveform.shape[-1] / float(TARGET_SAMPLE_RATE)
    if duration_seconds < MIN_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{stage}: audio too short ({duration_seconds:.3f}s) for file '{filename}'. "
                f"Minimum is {MIN_AUDIO_DURATION_SECONDS:.3f}s."
            ),
        )

    logger.info(
        "%s normalized audio for file=%s (channels=%s, sample_rate=%s, duration_seconds=%.3f)",
        stage,
        filename,
        waveform.shape[0],
        TARGET_SAMPLE_RATE,
        duration_seconds,
    )
    return waveform


def prepare_normalized_waveform(content: bytes, filename: str, stage: str) -> torch.Tensor:
    tmp_input_path = None
    try:
        suffix = resolve_audio_suffix(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            tmp_input.write(content)
            tmp_input_path = tmp_input.name

        waveform, sample_rate = load_audio_with_fallback(tmp_input_path)
        return validate_and_normalize_waveform(waveform, sample_rate, filename, stage)
    finally:
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)


def run_pipeline_on_waveform(
    pipeline: Pipeline, waveform: torch.Tensor, sample_rate: int
):
    """
    Run pyannote on an in-memory waveform to avoid file decoding through
    torchcodec/FFmpeg inside the container.
    """
    diarize_output = pipeline(
        {
            "waveform": waveform.contiguous(),
            "sample_rate": sample_rate,
        }
    )
    return getattr(diarize_output, "speaker_diarization", diarize_output)


def pad_waveform_to_full_second(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    expected_samples = ((waveform.shape[-1] // sample_rate) + 1) * sample_rate
    pad_length = expected_samples - waveform.shape[-1]
    if pad_length > 0:
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    return waveform


def build_speaker_segments(annotation) -> tuple[list[Dict[str, Any]], int]:
    speaker_segments = []
    speakers_set = set()

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 5),
                "end": round(turn.end, 5),
            }
        )
        speakers_set.add(speaker)

    return speaker_segments, len(speakers_set)


def diarize_audio_bytes(
    content: bytes,
    filename: str,
    requested_mode: str,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    try:
        target_device = resolve_device(requested_mode)
        logger.info(
            f"Diarization request received: file={filename}, requested_mode={requested_mode}, "
            f"resolved_device={target_device}, bytes={len(content)}"
        )
        pipeline = get_pipeline_for_device(target_device)
        waveform = prepare_normalized_waveform(content, filename, stage="diarization")
        waveform = pad_waveform_to_full_second(waveform, TARGET_SAMPLE_RATE)
        logger.info(
            f"Audio prepared for diarization: sample_rate={TARGET_SAMPLE_RATE}, "
            f"duration_seconds={round(waveform.shape[-1] / TARGET_SAMPLE_RATE, 3)}"
        )

        logger.info(f"Processing audio: {filename}")
        annotation = run_pipeline_on_waveform(pipeline, waveform, TARGET_SAMPLE_RATE)
        speaker_segments, num_speakers = build_speaker_segments(annotation)

        logger.info(
            f"Diarization complete: {len(speaker_segments)} segments, {num_speakers} speakers, "
            f"elapsed_seconds={round(time.perf_counter() - started_at, 3)}"
        )
        return {
            "analysis_type": "speaker_diarization",
            "results": {
                "speaker_segments": speaker_segments,
                "num_speakers": num_speakers,
            },
            "model_version": "diarization_v1",
            "audio_file": filename,
            "inference_device": str(target_device),
            "inference_mode": requested_mode,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Diarization failure for file=%s: %s", filename, exc)
        raise


def resample_waveform(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int
) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    return resampler(waveform)


def prepare_gender_waveform(content: bytes, filename: str) -> torch.Tensor:
    try:
        return prepare_normalized_waveform(content, filename, stage="gender")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Audio preparation for gender inference failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Gender classification failed: {exc}",
        ) from exc


def predict_gender_from_waveform(
    waveform: torch.Tensor,
    requested_mode: str,
) -> Dict[str, Any]:
    try:
        target_device = resolve_device(requested_mode)
        logger.info(
            f"Gender inference request: requested_mode={requested_mode}, resolved_device={target_device}, "
            f"audio_samples={waveform.shape[-1]}"
        )
        model = get_gender_model_for_device(target_device)
        waveform = waveform.to(target_device)

        with torch.no_grad():
            logits = model.forward(waveform)
            probs = torch.softmax(logits, dim=1).cpu()[0]
            pred_idx = int(logits.argmax(dim=1).item())

        return {
            "caller_gender": model.pred2gender[pred_idx],
            "confidence": round(float(probs[pred_idx].item()), 4),
            "model_version": "gender_classification_v1",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Local gender inference failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Gender classification failed: {exc}",
        ) from exc


def infer_gender_from_audio(
    content: bytes, filename: str, requested_mode: str
) -> Dict[str, Any]:
    waveform = prepare_gender_waveform(content, filename)
    return predict_gender_from_waveform(waveform, requested_mode)


def infer_speaker_genders(
    waveform: torch.Tensor,
    speaker_segments: list[Dict[str, Any]],
    requested_mode: str,
    sample_rate: int = 16000,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    speaker_chunks: Dict[str, list[torch.Tensor]] = {}

    for segment in speaker_segments:
        speaker = segment["speaker"]
        start = max(0, int(round(float(segment["start"]) * sample_rate)))
        end = min(waveform.shape[-1], int(round(float(segment["end"]) * sample_rate)))
        if end <= start:
            continue
        speaker_chunks.setdefault(speaker, []).append(waveform[:, start:end])

    speaker_genders = []
    speaker_gender_map: Dict[str, Dict[str, Any]] = {}

    for speaker in sorted(speaker_chunks):
        chunks = speaker_chunks[speaker]
        if not chunks:
            continue
        speaker_waveform = torch.cat(chunks, dim=-1)
        gender_result = predict_gender_from_waveform(speaker_waveform, requested_mode)
        speaker_entry = {
            "speaker": speaker,
            "gender": gender_result["caller_gender"],
            "confidence": gender_result["confidence"],
        }
        speaker_genders.append(speaker_entry)
        speaker_gender_map[speaker] = speaker_entry

    enriched_segments = []
    for segment in speaker_segments:
        speaker_entry = speaker_gender_map.get(segment["speaker"])
        enriched_segment = dict(segment)
        if speaker_entry is not None:
            enriched_segment["gender"] = speaker_entry["gender"]
            enriched_segment["confidence"] = speaker_entry["confidence"]
        enriched_segments.append(enriched_segment)

    return speaker_genders, enriched_segments

logger.info(f"Default inference device mode: {DEFAULT_INFERENCE_DEVICE}")
logger.info(f"HF_TOKEN present: {bool(token)}")
logger.info(f"Gender model configured: {bool(GENDER_MODEL_ID)}")
logger.info(f"Target normalized sample rate: {TARGET_SAMPLE_RATE}")
logger.info(f"Minimum accepted audio duration (seconds): {MIN_AUDIO_DURATION_SECONDS}")
log_runtime_device_details()

try:
    startup_device = resolve_device(DEFAULT_INFERENCE_DEVICE)
    get_pipeline_for_device(startup_device)
    get_gender_model_for_device(startup_device)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Speaker Diarization API",
    description="API for speaker diarization using PyAnnote",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "default_inference_device": DEFAULT_INFERENCE_DEVICE,
        "cuda_available": torch_cuda_available,
        "loaded_pipeline_devices": list(_pipeline_cache.keys()),
        "loaded_gender_model_devices": list(_gender_model_cache.keys()),
        "runtime_device": runtime_device_snapshot(),
    }


@app.post("/diarize")
async def diarize_audio(
    file: UploadFile = File(...),
    device: Literal["auto", "cpu", "gpu"] = Query(default="auto", description="Inference device mode")
) -> Dict[str, Any]:
    """
    Perform speaker diarization on uploaded audio file.
    
    Returns JSON with speaker segments, speaker count, and metadata.
    """
    try:
        requested_mode = device or DEFAULT_INFERENCE_DEVICE
        logger.info(
            "[runtime] /diarize request: requested_mode=%s, resolved_device=%s, cuda_available=%s",
            requested_mode,
            resolve_device(requested_mode),
            torch_cuda_available,
        )
        content = await file.read()
        logger.info(f"Endpoint /diarize called: filename={file.filename}, requested_mode={requested_mode}")
        return diarize_audio_bytes(content, file.filename, requested_mode)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")


@app.post("/diarize_gender")
async def diarize_audio_with_gender(
    file: UploadFile = File(...),
    device: Literal["auto", "cpu", "gpu"] = Query(default="auto", description="Inference device mode")
) -> Dict[str, Any]:
    """
    Perform speaker diarization and enrich the response with local gender
    classification.
    """
    try:
        requested_mode = device or DEFAULT_INFERENCE_DEVICE
        logger.info(
            "[runtime] /diarize_gender request: requested_mode=%s, resolved_device=%s, cuda_available=%s",
            requested_mode,
            resolve_device(requested_mode),
            torch_cuda_available,
        )
        content = await file.read()
        logger.info(
            f"Endpoint /diarize_gender called: filename={file.filename}, requested_mode={requested_mode}"
        )
        diarization_response = diarize_audio_bytes(content, file.filename, requested_mode)
        gender_waveform = prepare_gender_waveform(content, file.filename)
        speaker_genders, enriched_segments = infer_speaker_genders(
            gender_waveform,
            diarization_response["results"]["speaker_segments"],
            requested_mode,
        )
        gender_response = predict_gender_from_waveform(gender_waveform, requested_mode)

        diarization_response["analysis_type"] = "speaker_diarization_gender"
        diarization_response["results"]["speaker_segments"] = enriched_segments
        diarization_response["results"]["caller_gender"] = gender_response["caller_gender"]
        diarization_response["results"]["confidence"] = gender_response["confidence"]
        diarization_response["results"]["speaker_genders"] = speaker_genders
        diarization_response["model_version"] = {
            "diarization": diarization_response["model_version"],
            "gender": gender_response["model_version"],
        }
        return diarization_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diarization with gender failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diarization with gender failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=1200)
