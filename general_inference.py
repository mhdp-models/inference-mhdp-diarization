import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from app import (
    DEFAULT_INFERENCE_DEVICE,
    diarize_audio_bytes,
    infer_speaker_genders,
    predict_gender_from_waveform,
    prepare_gender_waveform,
)


def run_diarization_and_gender(audio_path: Path, device: str) -> Dict[str, Any]:
    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    content = audio_path.read_bytes()
    filename = audio_path.name

    diarization_response = diarize_audio_bytes(content, filename, device)
    gender_waveform = prepare_gender_waveform(content, filename)
    speaker_genders, enriched_segments = infer_speaker_genders(
        gender_waveform,
        diarization_response["results"]["speaker_segments"],
        device,
    )
    gender_response = predict_gender_from_waveform(gender_waveform, device)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization and gender classification on an audio file."
    )
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default=DEFAULT_INFERENCE_DEVICE,
        help="Inference device mode (default from INFERENCE_DEVICE env, fallback auto).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save JSON output. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)

    try:
        result = run_diarization_and_gender(audio_path, args.device)
    except HTTPException as exc:
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        raise SystemExit(f"Inference failed: {message}")
    except Exception as exc:
        raise SystemExit(f"Inference failed: {exc}")

    indent = 2 if args.pretty else None
    output_text = json.dumps(result, ensure_ascii=True, indent=indent)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + os.linesep, encoding="utf-8")
        print(f"Saved inference output to {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()