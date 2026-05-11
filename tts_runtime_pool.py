from __future__ import annotations

import asyncio
import gc
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

from infer import InferenceRuntime, RuntimeKey

T = TypeVar("T")


@dataclass
class TTSWorkerState:
    index: int
    device: str
    runtime: InferenceRuntime | None = None
    model_id: str | None = None
    checkpoint: str | None = None
    loaded: bool = False
    busy: bool = False
    last_used_at: float | None = None
    load_started_at: float | None = None
    load_finished_at: float | None = None
    current_request_id: str | None = None
    last_error: str | None = None
    _load_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class TTSWorkerPool:
    """Small async worker pool for lazy Irodori-TTS runtime reuse."""

    def __init__(
        self,
        *,
        runtime_key_for: Callable[[str], RuntimeKey],
        num_workers: int = 1,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        self._runtime_key_for = runtime_key_for
        self._num_workers = int(num_workers)
        self._log_fn = log_fn or (lambda message: print(message, flush=True))
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_workers,
            thread_name_prefix="tts-worker",
        )
        self._workers = [
            TTSWorkerState(index=index, device="unknown") for index in range(self._num_workers)
        ]
        self._available: asyncio.Queue[TTSWorkerState] = asyncio.Queue()
        for worker in self._workers:
            self._available.put_nowait(worker)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def is_loaded(self) -> bool:
        return any(worker.loaded for worker in self._workers)

    async def synthesize(
        self,
        *,
        model_id: str,
        request_id: str,
        work: Callable[[InferenceRuntime], T],
    ) -> T:
        worker = await self._available.get()
        worker.busy = True
        worker.current_request_id = request_id
        worker.last_error = None
        worker.last_used_at = time.monotonic()
        started_at = time.perf_counter()

        try:
            runtime = await self._ensure_loaded(worker, model_id, request_id=request_id)
            device = str(runtime.model_device)
            self._log(
                f"[tts:pool] synth_start id={request_id} worker={worker.index} "
                f"model={model_id} device={device}"
            )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, work, runtime)
        except Exception:
            worker.last_error = traceback.format_exc()
            self._log(f"[tts:pool] error id={request_id}\n{worker.last_error}")
            raise
        finally:
            elapsed = time.perf_counter() - started_at
            worker.busy = False
            worker.current_request_id = None
            worker.last_used_at = time.monotonic()
            self._available.put_nowait(worker)
            self._log(
                f"[tts:pool] synth_end id={request_id} worker={worker.index} "
                f"elapsed={elapsed:.2f}s available={self._available.qsize()}/{self._num_workers}"
            )

    async def unload_idle_models(self, *, force: bool = False) -> int:
        if not force:
            return 0

        unloaded = 0
        for worker in self._workers:
            async with worker._load_lock:
                if worker.busy or worker.runtime is None:
                    continue
                self._unload_worker(worker, reason="force")
                unloaded += 1
        return unloaded

    def get_health(self) -> dict:
        now = time.monotonic()
        loaded_workers = [worker for worker in self._workers if worker.loaded]
        primary = loaded_workers[0] if loaded_workers else self._workers[0]
        idle_seconds = None
        if primary.last_used_at is not None and not primary.busy:
            idle_seconds = max(0.0, now - primary.last_used_at)

        return {
            "status": "ok",
            "loaded": bool(loaded_workers),
            "busy": any(worker.busy for worker in self._workers),
            "queue_size": self._num_workers - self._available.qsize(),
            "num_workers": self._num_workers,
            "available_workers": self._available.qsize(),
            "device": primary.device,
            "idle_seconds": idle_seconds,
            "model": primary.model_id,
            "workers": [
                {
                    "index": worker.index,
                    "loaded": worker.loaded,
                    "busy": worker.busy,
                    "device": worker.device,
                    "model": worker.model_id,
                    "checkpoint": worker.checkpoint,
                    "last_used_at": worker.last_used_at,
                    "load_started_at": worker.load_started_at,
                    "load_finished_at": worker.load_finished_at,
                    "current_request_id": worker.current_request_id,
                }
                for worker in self._workers
            ],
        }

    async def unload_all_models(self) -> int:
        return await self.unload_idle_models(force=True)

    def shutdown_executor(self) -> None:
        self._executor.shutdown(wait=True)

    async def shutdown(self) -> None:
        await self.unload_all_models()
        self.shutdown_executor()

    async def _ensure_loaded(
        self,
        worker: TTSWorkerState,
        model_id: str,
        *,
        request_id: str | None,
    ) -> InferenceRuntime:
        async with worker._load_lock:
            if worker.runtime is not None and worker.model_id == model_id:
                worker.loaded = True
                worker.last_used_at = time.monotonic()
                return worker.runtime

            if worker.runtime is not None:
                self._unload_worker(worker, reason="switch")

            key = self._runtime_key_for(model_id)
            worker.load_started_at = time.monotonic()
            worker.load_finished_at = None
            worker.model_id = model_id
            worker.checkpoint = key.checkpoint
            worker.device = key.model_device

            load_start = time.perf_counter()
            self._log(
                f"[tts:pool] model_load start id={request_id} worker={worker.index} "
                f"model={model_id} checkpoint={key.checkpoint} device={key.model_device}"
            )
            loop = asyncio.get_running_loop()
            runtime = await loop.run_in_executor(
                self._executor,
                InferenceRuntime.from_key,
                key,
            )

            worker.runtime = runtime
            worker.loaded = True
            worker.device = str(runtime.model_device)
            worker.last_used_at = time.monotonic()
            worker.load_finished_at = worker.last_used_at
            elapsed = time.perf_counter() - load_start
            self._log(
                f"[tts:pool] model_load end id={request_id} worker={worker.index} "
                f"model={model_id} elapsed={elapsed:.2f}s device={worker.device} "
                f"use_speaker_condition={runtime.model_cfg.use_speaker_condition} "
                f"use_caption_condition={runtime.model_cfg.use_caption_condition}"
            )
            return runtime

    def _unload_worker(self, worker: TTSWorkerState, *, reason: str) -> None:
        runtime = worker.runtime
        if runtime is None:
            return

        model_id = worker.model_id
        idle_seconds = None
        if worker.last_used_at is not None:
            idle_seconds = max(0.0, time.monotonic() - worker.last_used_at)

        self._log(
            f"[tts:pool] model_unload worker={worker.index} model={model_id} "
            f"reason={reason} idle_seconds={idle_seconds}"
        )
        try:
            runtime.unload()
        finally:
            worker.runtime = None
            worker.model_id = None
            worker.checkpoint = None
            worker.loaded = False
            worker.load_started_at = None
            worker.load_finished_at = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                self._log(f"[tts:pool] cuda_empty_cache skipped error={exc}")

    def _log(self, message: str) -> None:
        self._log_fn(message)
