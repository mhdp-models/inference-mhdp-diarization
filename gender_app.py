from pathlib import Path
import tempfile
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from model import ECAPA_gender

ANALYSIS_TYPE = "gender_classification"
MODEL_VERSION = "gender_classification_v1"
MODEL_ID = os.getenv("GENDER_MODEL_ID", "mhdp-africa/gender_classification_MHDP_asr_dataset_V1")

app = FastAPI(title="Gender Classification API", version=MODEL_VERSION)


def _load_model() -> tuple[ECAPA_gender, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECAPA_gender.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)
    return model, device


MODEL, DEVICE = _load_model()


class PathInferenceRequest(BaseModel):
    audio_path: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "analysis_type": ANALYSIS_TYPE,
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
    }


def _predict_from_path(audio_path: str) -> dict:
    audio = MODEL.load_audio(audio_path).to(DEVICE)

    with torch.no_grad():
        logits = MODEL.forward(audio)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())

    return {
        "analysis_type": ANALYSIS_TYPE,
        "results": {
            "caller_gender": MODEL.pred2gender[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
        },
        "model_version": MODEL_VERSION,
    }


@app.post("/infer/gender")
async def infer_gender(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        return _predict_from_path(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/infer/gender/path")
def infer_gender_path(payload: PathInferenceRequest) -> dict:
    audio_path = payload.audio_path

    if not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    try:
        return _predict_from_path(audio_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc