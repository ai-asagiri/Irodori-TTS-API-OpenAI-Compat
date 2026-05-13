from __future__ import annotations

import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from config import CONFIG

if str(CONFIG.irodori_root) not in sys.path:
    sys.path.insert(0, str(CONFIG.irodori_root))

from infer import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    hf_hub_download,
    resolve_cfg_scales,
    save_wav,
)
from tts_runtime_pool import TTSWorkerPool

BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "refs"
READING_REPLACEMENTS_PATH = BASE_DIR / "reading_replacements.json"

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_MODEL = CONFIG.default_model
DEFAULT_VOICE = CONFIG.default_voice
MAX_CHUNK_CHARS = 60
MERGE_SHORT_SENTENCE_MAX_CHARS = 45
MIN_CHUNK_CHARS = 20
SILENCE_BETWEEN_CHUNKS_SECONDS = CONFIG.chunk_silence_seconds
CLOSING_BRACKET_CHARS = "」』）)”】〕〉》］｝"
BRACKET_CHARS = "「『（(［[｛{【〔〈《」』）)］]｝}】〕〉》“”\"'"

# Generated wav files are kept here and pruned by total size.
AUDIO_OUTPUT_DIR = CONFIG.output_dir
AUDIO_DELETE_DELAY_SECONDS = CONFIG.delete_delay_seconds
AUDIO_OUTPUT_MAX_BYTES = CONFIG.audio_output_max_bytes
# Never prune below this count, even if the directory is over the byte limit.
AUDIO_OUTPUT_MIN_KEEP_FILES = 3


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    use_speaker_condition: bool
    use_caption_condition: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "irodori-tts": ModelSpec(
        repo_id="Aratako/Irodori-TTS-500M-v2",
        use_speaker_condition=True,
        use_caption_condition=False,
    ),
    "irodori-tts-voice-design": ModelSpec(
        repo_id="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        use_speaker_condition=False,
        use_caption_condition=True,
    ),
}
_runtime_key_cache: dict[str, RuntimeKey] = {}
_runtime_key_cache_lock = Lock()


class CommonParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int | None = 42
    num_steps: int = 24
    cfg_scale_text: float = 2.0
    cfg_guidance_mode: str = "independent"
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1


class DefaultModelParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cfg_scale_speaker: float = 5.0
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    max_ref_seconds: float | None = 30.0


class VoiceDesignModelParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    caption: str | None = None
    cfg_scale_caption: float = 1.0


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = DEFAULT_MODEL
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float | None = None

    common: CommonParams = Field(default_factory=CommonParams)

    irodori_tts: DefaultModelParams = Field(
        default_factory=DefaultModelParams,
        alias="irodori-tts",
    )
    irodori_tts_voice_design: VoiceDesignModelParams = Field(
        default_factory=VoiceDesignModelParams,
        alias="irodori-tts-voice-design",
    )


app = FastAPI(title="Irodori-TTS OpenAI Compatible API")


def load_voice_refs() -> dict[str, Path]:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    return {wav.stem: wav for wav in sorted(REF_DIR.glob("*.wav"))}


def visible_voice_names() -> list[str]:
    return list(load_voice_refs().keys())


def clean_voice(value: str | None) -> str:
    return (value or "").strip()


def resolve_reference_voice(voice_name: str) -> Path | None:
    if not voice_name:
        return None

    voices = load_voice_refs()
    ref_path = voices.get(voice_name)
    if ref_path is None:
        available = ", ".join(visible_voice_names()) or "(none)"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice_name}. Available voices: {available}",
        )
    if not ref_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Reference wav not found for voice: {voice_name}",
        )

    return ref_path


def model_items() -> list[dict[str, str]]:
    return [
        {
            "id": model_id,
            "object": "model",
            "name": model_id,
            "owned_by": "local",
        }
        for model_id in MODEL_SPECS.keys()
    ]


def runtime_key_for(model_id: str) -> RuntimeKey:
    with _runtime_key_cache_lock:
        cached = _runtime_key_cache.get(model_id)
        if cached is not None:
            return cached

        model_spec = MODEL_SPECS[model_id]
        checkpoint_path = hf_hub_download(
            repo_id=model_spec.repo_id,
            filename="model.safetensors",
        )

        device = default_runtime_device()
        key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=device,
            codec_repo=CODEC_REPO,
            model_precision="bf16",
            codec_device=device,
            codec_precision="bf16",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            enable_watermark=False,
            compile_model=False,
            compile_dynamic=False,
        )
        _runtime_key_cache[model_id] = key
        return key


tts_pool = TTSWorkerPool(
    runtime_key_for=runtime_key_for,
    num_workers=CONFIG.num_workers,
    log_fn=lambda message: print(message, flush=True),
)


def force_split_long_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    comma_pattern = re.compile(r"[、，,]")

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1
        for match in comma_pattern.finditer(window):
            cut = match.end()

        if cut < MIN_CHUNK_CHARS:
            cut = max_chars

        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def load_reading_replacements() -> dict[str, str]:
    if not READING_REPLACEMENTS_PATH.exists():
        return {}

    with READING_REPLACEMENTS_PATH.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"reading replacements must be a JSON object: {READING_REPLACEMENTS_PATH}"
        )

    replacements: dict[str, str] = {}
    for source, replacement in loaded.items():
        source_text = str(source)
        if source_text:
            replacements[source_text] = str(replacement)
    return replacements


def split_text_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []

    parts = re.split(
        rf"(\r?\n+|[。！？!?]+[{re.escape(CLOSING_BRACKET_CHARS)}]*)",
        text,
    )
    chunks: list[str] = []
    current = ""

    for index in range(0, len(parts), 2):
        body = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        sentence = apply_reading_replacements((body + delimiter).strip())
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(force_split_long_text(sentence, max_chars))
            continue

        candidate = f"{current}{sentence}" if current else sentence
        if len(candidate) <= MERGE_SHORT_SENTENCE_MAX_CHARS:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = sentence

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]


def apply_reading_replacements(text: str) -> str:
    for source, replacement in load_reading_replacements().items():
        text = text.replace(source, replacement)
    return text


ELLIPSIS_RE = re.compile(r"(?:…+|\.{3,}|・・・+|･･･+)")


def count_ellipsis(text: str) -> int:
    return len(ELLIPSIS_RE.findall(text))


def estimate_speech_units(text: str) -> float:
    body = ELLIPSIS_RE.sub("", text)
    body = body.translate(str.maketrans("", "", BRACKET_CHARS))

    units = 0.0
    for ch in body:
        if re.match(r"[、，,。！？!?\s]", ch):
            continue
        if ch in "ゃゅょャュョぁぃぅぇぉァィゥェォ":
            continue
        if ch in "っッー":
            units += 1.0
        elif re.match(r"[\u3400-\u9fff々〆ヵヶ]", ch):
            units += 1.4
        elif re.match(r"[\u3041-\u3096\u30A1-\u30FA]", ch):
            units += 1.0
        elif ch.isascii() and ch.isalnum():
            units += 0.7
        else:
            units += 1.0

    return units


def seconds_for_chunk(text: str) -> float:
    text = apply_reading_replacements(text)
    units = estimate_speech_units(text)
    comma_count = len(re.findall(r"[、，,]", text))
    sentence_end_count = len(re.findall(r"[。！？!?]", text))
    ellipsis_count = count_ellipsis(text)

    seconds = (
        units * 0.12
        + comma_count * 0.60
        + sentence_end_count * 0.25
        + ellipsis_count * 0.80
        + 1.00
    )

    return max(4.6, min(30.0, seconds))


def audio_to_numpy_float32(audio) -> np.ndarray:
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return audio.detach().to(dtype=torch.float32).cpu().numpy()
    except Exception:
        pass

    return np.asarray(audio, dtype=np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    normalized = np.asarray(audio)

    if normalized.ndim == 1:
        return normalized.astype(np.float32, copy=False)

    if normalized.ndim == 2:
        if normalized.shape[0] <= 8 and normalized.shape[1] > normalized.shape[0]:
            return normalized.mean(axis=0).astype(np.float32, copy=False)
        if normalized.shape[1] <= 8:
            return normalized.mean(axis=1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported audio shape: {normalized.shape}")


def silence_audio(
    sample_rate: int,
    dtype: np.dtype,
    seconds: float = SILENCE_BETWEEN_CHUNKS_SECONDS,
) -> np.ndarray:
    samples = int(sample_rate * seconds)
    return np.zeros(samples, dtype=dtype)


def create_audio_output_path() -> Path:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_OUTPUT_DIR / f"speech_{uuid.uuid4().hex}.wav"


def delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        print(f"[tts] cleanup deleted path={path}", flush=True)
    except Exception as e:
        print(f"[tts] cleanup failed path={path} error={e}", flush=True)


def delete_file_later(path: Path, delay_seconds: float = AUDIO_DELETE_DELAY_SECONDS) -> None:
    try:
        time.sleep(delay_seconds)
        path.unlink(missing_ok=True)
        print(f"[tts] deleted temporary audio file: {path}", flush=True)
    except Exception as e:
        print(f"[tts] failed to delayed delete file: {path} error={e}", flush=True)


def prune_audio_output_dir() -> None:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [path for path in AUDIO_OUTPUT_DIR.glob("*.wav") if path.is_file()]
    if len(files) <= AUDIO_OUTPUT_MIN_KEEP_FILES:
        return

    def file_stat(path: Path):
        try:
            stat = path.stat()
            return stat.st_mtime, stat.st_size
        except OSError:
            return 0.0, 0

    file_infos = [(path, *file_stat(path)) for path in files]
    total_bytes = sum(size for _, _, size in file_infos)

    if total_bytes <= AUDIO_OUTPUT_MAX_BYTES:
        return

    deleted_count = 0
    file_infos.sort(key=lambda item: item[1])  # oldest first

    for path, _, size in file_infos:
        if len(file_infos) - deleted_count <= AUDIO_OUTPUT_MIN_KEEP_FILES:
            break
        if total_bytes <= AUDIO_OUTPUT_MAX_BYTES:
            break

        try:
            path.unlink(missing_ok=True)
            total_bytes -= size
            deleted_count += 1
            print(
                f"[tts] pruned audio file: {path} remaining_bytes={total_bytes}",
                flush=True,
            )
        except Exception as e:
            print(f"[tts] failed to prune audio file: {path} error={e}", flush=True)


def model_log_kind(model_id: str) -> str:
    if model_id == "irodori-tts-voice-design":
        return "voice-design"
    return "default"


def clean_caption(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@app.on_event("shutdown")
async def shutdown_tts_pool() -> None:
    await shutdown_tts_resources()


async def shutdown_tts_resources() -> None:
    await tts_pool.shutdown()


@app.get("/health")
def health():
    pool_health = tts_pool.get_health()
    return {
        **pool_health,
        "models": list(MODEL_SPECS.keys()),
        "loaded_model": pool_health.get("model"),
        "voices": visible_voice_names(),
        "audio_output_dir": str(AUDIO_OUTPUT_DIR),
        "audio_output_max_bytes": AUDIO_OUTPUT_MAX_BYTES,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": model_items(),
    }


@app.get("/v1/audio/models")
def list_audio_models():
    items = model_items()
    return {
        "object": "list",
        "data": items,
        "models": [{"id": item["id"], "name": item["name"]} for item in items],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "voice",
                "name": name,
            }
            for name in visible_voice_names()
        ],
    }


@app.get("/v1/audio/voices")
def list_audio_voices():
    return {
        "voices": [
            {
                "id": name,
                "name": name,
            }
            for name in visible_voice_names()
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    model_id = (req.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    log_kind = model_log_kind(model_id)
    request_id = uuid.uuid4().hex[:8]
    start_time = time.perf_counter()

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input is empty")

    if (req.response_format or "wav").lower() != "wav":
        raise HTTPException(status_code=400, detail="Only wav response_format is supported")

    if model_id not in MODEL_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}. Available models: {', '.join(MODEL_SPECS.keys())}",
        )

    common = req.common
    default_params = req.irodori_tts
    voice_design_params = req.irodori_tts_voice_design

    model_spec = MODEL_SPECS[model_id]
    use_speaker_condition = model_spec.use_speaker_condition
    use_caption_model = model_spec.use_caption_condition

    caption = clean_caption(voice_design_params.caption) if use_caption_model else None
    use_caption_condition = bool(use_caption_model and caption)

    if use_caption_model and not caption:
        raise HTTPException(
            status_code=400,
            detail='VoiceDesign model requires "irodori-tts-voice-design.caption" in Additional Parameters',
        )

    ref_path: Path | None = None
    requested_voice_name = clean_voice(req.voice)
    voice_name = requested_voice_name

    if use_speaker_condition:
        voice_name = requested_voice_name or DEFAULT_VOICE
        ref_path = resolve_reference_voice(voice_name)
        if ref_path is None:
            voice_name = ""
    else:
        voice_name = voice_name or "ignored"

    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode=common.cfg_guidance_mode,
        cfg_scale_text=float(common.cfg_scale_text),
        cfg_scale_caption=float(voice_design_params.cfg_scale_caption),
        cfg_scale_speaker=float(default_params.cfg_scale_speaker),
        cfg_scale=None,
        use_caption_condition=use_caption_condition,
        use_speaker_condition=use_speaker_condition,
    )

    chunks = split_text_for_tts(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="input is empty after chunk split")

    no_ref = ref_path is None

    if req.speed is not None and req.speed != 1.0:
        print(
            f"[tts:{log_kind}] unsupported_speed "
            f"id={request_id} "
            f"speed={req.speed} ignored=True",
            flush=True,
        )

    print(
        f"[tts:{log_kind}] input "
        f"id={request_id} "
        f"chars={len(text)} "
        f"repr={text!r}",
        flush=True,
    )
    print(
        f"[tts:{log_kind}] request "
        f"id={request_id} "
        f"model={model_id} voice={voice_name} ref_path={ref_path} "
        f"no_ref={no_ref} "
        f"use_speaker_condition={use_speaker_condition} "
        f"use_caption_condition={use_caption_condition} "
        f"caption={caption!r}"
    )
    print(
        f"[tts:{log_kind}] guidance "
        f"id={request_id} "
        f"cfg_scale_text={cfg_scale_text} "
        f"cfg_scale_caption={cfg_scale_caption} "
        f"cfg_scale_speaker={cfg_scale_speaker} "
        f"cfg_guidance_mode={common.cfg_guidance_mode} "
        f"seed={common.seed} "
        f"num_steps={common.num_steps}"
    )
    print(
        f"[tts:{log_kind}] split "
        f"id={request_id} "
        f"chunks={len(chunks)} "
        f"lengths={[len(chunk) for chunk in chunks]}",
        flush=True,
    )

    prune_audio_output_dir()
    out_path = create_audio_output_path()

    def run_synthesis(runtime: InferenceRuntime) -> Path:
        audios: list[np.ndarray] = []
        sample_rate: int | None = None

        for index, chunk in enumerate(chunks, start=1):
            chunk_seconds = seconds_for_chunk(chunk)
            chunk_start = time.perf_counter()
            print(
                f"[tts:{log_kind}] chunk "
                f"id={request_id} "
                f"index={index}/{len(chunks)} "
                f"chars={len(chunk)} "
                f"seconds={chunk_seconds:.2f} "
                f"repr={chunk!r}",
                flush=True,
            )
            result = runtime.synthesize(
                SamplingRequest(
                    text=chunk,
                    caption=caption,
                    ref_wav=str(ref_path) if ref_path is not None else None,
                    ref_latent=None,
                    no_ref=no_ref,
                    ref_normalize_db=default_params.ref_normalize_db,
                    ref_ensure_max=bool(default_params.ref_ensure_max),
                    num_candidates=1,
                    decode_mode="sequential",
                    seconds=chunk_seconds,
                    max_ref_seconds=default_params.max_ref_seconds,
                    max_text_len=None,
                    max_caption_len=None,
                    num_steps=int(common.num_steps),
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_caption=cfg_scale_caption,
                    cfg_scale_speaker=cfg_scale_speaker,
                    cfg_guidance_mode=common.cfg_guidance_mode,
                    cfg_scale=None,
                    cfg_min_t=0.5,
                    cfg_max_t=1.0,
                    truncation_factor=None,
                    rescale_k=None,
                    rescale_sigma=None,
                    context_kv_cache=True,
                    speaker_kv_scale=None,
                    speaker_kv_min_t=None,
                    speaker_kv_max_layers=None,
                    seed=common.seed,
                    trim_tail=bool(common.trim_tail),
                    tail_window_size=int(common.tail_window_size),
                    tail_std_threshold=float(common.tail_std_threshold),
                    tail_mean_threshold=float(common.tail_mean_threshold),
                ),
                log_fn=None,
            )

            if sample_rate is None:
                sample_rate = int(result.sample_rate)
            elif sample_rate != int(result.sample_rate):
                raise ValueError(
                    f"Chunk sample_rate mismatch: first={sample_rate} current={result.sample_rate}"
                )

            raw_audio = audio_to_numpy_float32(result.audio)
            normalized_audio = normalize_audio(raw_audio)
            print(
                f"[tts:{log_kind}] audio_shape "
                f"id={request_id} "
                f"index={index}/{len(chunks)} "
                f"raw_shape={raw_audio.shape} "
                f"normalized_shape={normalized_audio.shape} "
                f"dtype={normalized_audio.dtype} "
                f"sr={sample_rate}",
                flush=True,
            )

            audios.append(normalized_audio)
            if index < len(chunks):
                audios.append(silence_audio(sample_rate, normalized_audio.dtype))

            chunk_elapsed = time.perf_counter() - chunk_start
            print(
                f"[tts:{log_kind}] chunk_done "
                f"id={request_id} "
                f"index={index}/{len(chunks)} "
                f"elapsed={chunk_elapsed:.2f}s",
                flush=True,
            )

        if not audios or sample_rate is None:
            raise ValueError("No audio was generated")

        merged_audio = np.concatenate(audios, axis=0)
        if merged_audio.ndim == 1:
            merged_audio = np.expand_dims(merged_audio, axis=0)
        print(
            f"[tts:{log_kind}] merged_audio "
            f"id={request_id} "
            f"shape={merged_audio.shape} "
            f"dtype={merged_audio.dtype}",
            flush=True,
        )
        try:
            import torch

            merged_audio = torch.from_numpy(
                np.ascontiguousarray(merged_audio)
            ).to(dtype=torch.float32)
        except Exception:
            pass
        print(
            f"[tts:{log_kind}] merged_audio_tensor "
            f"id={request_id} "
            f"type={type(merged_audio)} "
            f"shape={tuple(merged_audio.shape)}",
            flush=True,
        )
        save_wav(str(out_path), merged_audio, sample_rate)
        elapsed = time.perf_counter() - start_time
        file_size = out_path.stat().st_size if out_path.exists() else 0
        print(
            f"[tts:{log_kind}] done "
            f"id={request_id} "
            f"chunks={len(chunks)} "
            f"elapsed={elapsed:.2f}s "
            f"bytes={file_size} "
            f"path={out_path}",
            flush=True,
        )
        return out_path

    try:
        await tts_pool.synthesize(
            model_id=model_id,
            request_id=request_id,
            work=run_synthesis,
        )
    except Exception as e:
        delete_file(out_path)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        path=str(out_path),
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(delete_file_later, out_path),
    )
