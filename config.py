from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = API_ROOT / "config.json"


@dataclass(frozen=True)
class AppConfig:
    irodori_root: Path
    output_dir: Path
    delete_delay_seconds: float
    chunk_silence_seconds: float
    num_workers: int
    audio_output_max_bytes: int
    default_model: str
    default_voice: str
    reading_replacements_path: Path


DEFAULT_CONFIG: dict[str, Any] = {
    "irodori_root": "../Irodori-TTS",
    "output_dir": "outputs/generated_audio",
    "delete_delay_seconds": 60,
    "chunk_silence_seconds": 0.45,
    "num_workers": 1,
    "audio_output_max_bytes": 2147483648,
    "default_model": "irodori-tts",
    "default_voice": "",
    "reading_replacements_path": "reading_replacements.json",
}


def _load_raw_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError(f"config.json must contain a JSON object: {CONFIG_PATH}")

    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config() -> AppConfig:
    raw = _load_raw_config()

    num_workers = int(raw["num_workers"])
    if num_workers < 1:
        raise ValueError("config num_workers must be >= 1")

    audio_output_max_bytes = int(raw["audio_output_max_bytes"])
    if audio_output_max_bytes < 0:
        raise ValueError("config audio_output_max_bytes must be >= 0")

    delete_delay_seconds = float(raw["delete_delay_seconds"])
    if delete_delay_seconds < 0:
        raise ValueError("config delete_delay_seconds must be >= 0")

    chunk_silence_seconds = float(raw["chunk_silence_seconds"])
    if chunk_silence_seconds < 0:
        raise ValueError("config chunk_silence_seconds must be >= 0")

    return AppConfig(
        irodori_root=_resolve_path(str(raw["irodori_root"]), base_dir=API_ROOT),
        output_dir=_resolve_path(str(raw["output_dir"]), base_dir=API_ROOT),
        delete_delay_seconds=delete_delay_seconds,
        chunk_silence_seconds=chunk_silence_seconds,
        num_workers=num_workers,
        audio_output_max_bytes=audio_output_max_bytes,
        default_model=str(raw["default_model"]).strip() or DEFAULT_CONFIG["default_model"],
        default_voice=str(raw["default_voice"]).strip(),
        reading_replacements_path=_resolve_path(
            str(raw["reading_replacements_path"]),
            base_dir=API_ROOT,
        ),
    )


CONFIG = load_config()
