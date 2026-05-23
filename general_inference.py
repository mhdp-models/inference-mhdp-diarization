import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from fastapi import HTTPException
from dotenv import load_dotenv


def load_inference_components(
    hf_token: str | None,
) -> Tuple[
    str,
    Callable[[bytes, str, str], Dict[str, Any]],
    Callable[[bytes, str], Any],
    Callable[[Any, list[Dict[str, Any]], str], tuple[list[Dict[str, Any]], list[Dict[str, Any]]]],
    Callable[[Any, str], Dict[str, Any]],
]:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    from app import (  # Local import ensures HF_TOKEN is set before app startup model loading.
        DEFAULT_INFERENCE_DEVICE,
        diarize_audio_bytes,
        infer_speaker_genders,
        predict_gender_from_waveform,
        prepare_gender_waveform,
    )

    return (
        DEFAULT_INFERENCE_DEVICE,
        diarize_audio_bytes,
        prepare_gender_waveform,
        infer_speaker_genders,
        predict_gender_from_waveform,
    )


def run_diarization_and_gender(
    audio_path: Path,
    device: str,
    diarize_audio_bytes: Callable[[bytes, str, str], Dict[str, Any]],
    prepare_gender_waveform: Callable[[bytes, str], Any],
    infer_speaker_genders: Callable[[Any, list[Dict[str, Any]], str], tuple[list[Dict[str, Any]], list[Dict[str, Any]]]],
    predict_gender_from_waveform: Callable[[Any, str], Dict[str, Any]],
) -> Dict[str, Any]:
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
        default="auto",
        help="Inference device mode (default: auto).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If omitted, uses HF_TOKEN from environment.",
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
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")

    args = parse_args()
    audio_path = Path(args.audio_path)

    resolved_hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not resolved_hf_token:
        raise SystemExit(
            "Inference failed: missing Hugging Face token. Set HF_TOKEN in your environment "
            "or pass --hf-token."
        )

    (
        default_inference_device,
        diarize_audio_bytes,
        prepare_gender_waveform,
        infer_speaker_genders,
        predict_gender_from_waveform,
    ) = load_inference_components(resolved_hf_token)

    requested_device = args.device or default_inference_device

    try:
        result = run_diarization_and_gender(
            audio_path,
            requested_device,
            diarize_audio_bytes,
            prepare_gender_waveform,
            infer_speaker_genders,
            predict_gender_from_waveform,
        )
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