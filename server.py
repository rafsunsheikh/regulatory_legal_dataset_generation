"""FastAPI server exposing a web UI and upload API for the PDF pipeline."""

import logging
import os
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure local imports work when running `uvicorn server:app`
sys.path.insert(0, str(Path(__file__).parent))

import psutil

from config import INSTRUCTION_PROMPT, OLLAMA_MODEL, OUTPUT_DATASET_FILE, RAW_PDF_DIR
from src.generator import test_ollama_connection
from src.pipeline import append_dataset_entries, ensure_processed_dir, process_pdf

logger = logging.getLogger(__name__)

app = FastAPI(title="Legal PDF Instruction Dataset Pipeline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static web UI
web_dir = Path("web")
web_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=web_dir, html=True), name="static")


class TaskStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskRecord:
    id: str
    filename: str
    model: str
    device: str
    prompt: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    temperature: Optional[float] = None
    max_retries: Optional[int] = None
    status: str = TaskStatus.QUEUED
    progress: float = 0.0
    generated: int = 0
    total_chunks: int = 0
    message: str = ""
    error: Optional[str] = None
    output_path: Optional[str] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    eta_seconds: Optional[float] = None
    resource_cpu: Optional[float] = None  # percent
    resource_mem_mb: Optional[float] = None


tasks: Dict[str, TaskRecord] = {}
tasks_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=3)
PROCESS = psutil.Process(os.getpid())


def _update_task(task_id: str, **kwargs) -> None:
    with tasks_lock:
        if task_id not in tasks:
            return
        for key, value in kwargs.items():
            setattr(tasks[task_id], key, value)


def _register_task(
    filename: str,
    model: str,
    device: str,
    prompt: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    output_path: Optional[str] = None,
) -> TaskRecord:
    task_id = uuid.uuid4().hex
    record = TaskRecord(
        id=task_id,
        filename=filename,
        model=model,
        device=device,
        prompt=prompt,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
        max_retries=max_retries,
        output_path=output_path,
    )
    with tasks_lock:
        tasks[task_id] = record
    return record


def _resource_snapshot() -> Dict[str, float]:
    """Return current process CPU% (since last call) and RSS in MB."""
    try:
        cpu = PROCESS.cpu_percent(interval=None)
        mem_mb = PROCESS.memory_info().rss / (1024 * 1024)
        return {"cpu": cpu, "mem_mb": mem_mb}
    except Exception:
        return {}


def _to_int(value, field: str) -> Optional[int]:
    if value in (None, "", []):
        return None
    try:
        return int(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid integer for {field}")


def _to_float(value, field: str) -> Optional[float]:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid float for {field}")


def _progress_callback(task_id: str):
    def _callback(event: Dict) -> None:
        stage = event.get("stage", "")
        with tasks_lock:
            task = tasks.get(task_id)
        if not task:
            return
        if stage == "chunked":
            total = event.get("total_chunks", 0)
            _update_task(
                task_id,
                total_chunks=total,
                message="Chunked document",
                updated_at=time.time(),
            )
        elif stage == "generating":
            total = event.get("total_chunks") or task.total_chunks or 1
            idx = event.get("chunk_index", 0)
            progress = min(1.0, max(0.0, (idx + 1) / total))
            now = time.time()
            started_at = task.started_at or now
            elapsed = max(0.0, now - started_at)
            eta = None
            if progress > 0:
                eta = max(0.0, (elapsed / progress) - elapsed)
            resources = _resource_snapshot()
            _update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=progress,
                eta_seconds=eta,
                resource_cpu=resources.get("cpu"),
                resource_mem_mb=resources.get("mem_mb"),
                message=f"Generating chunk {idx + 1}/{total} (model: {task.model})",
                updated_at=now,
            )
        elif stage == "completed":
            generated = event.get("generated", task.generated)
            _update_task(
                task_id,
                generated=generated,
                progress=1.0,
                eta_seconds=0.0,
                message="Generation complete",
                updated_at=time.time(),
            )
        elif stage == "failed":
            _update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=0.0,
                error=event.get("reason", "failed"),
                eta_seconds=None,
                updated_at=time.time(),
            )

    return _callback


def _dataset_path() -> Path:
    return Path(OUTPUT_DATASET_FILE)


def _dataset_summary() -> Dict:
    """Compute lightweight stats over the dataset JSONL."""
    path = _dataset_path()
    if not path.exists():
        return {
            "total": 0,
            "by_source": {},
            "by_model": {},
            "by_device": {},
            "by_prompt": {},
            "avg_instruction_chars": 0,
        }

    by_source: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_device: Counter[str] = Counter()
    by_prompt: Counter[str] = Counter()
    total = 0
    instr_chars = 0

    with jsonlines.open(path, mode="r") as reader:
        for row in reader:
            total += 1
            by_source[row.get("source_file", "unknown")] += 1
            by_model[row.get("model", row.get("ollama_model", "unknown"))] += 1
            by_device[row.get("device", "unknown")] += 1
            if row.get("prompt"):
                by_prompt["custom"] += 1
            instr_chars += len(str(row.get("instruction", "")))

    avg_instr = instr_chars / total if total else 0

    return {
        "total": total,
        "by_source": dict(by_source),
        "by_model": dict(by_model),
        "by_device": dict(by_device),
        "by_prompt": dict(by_prompt),
        "avg_instruction_chars": avg_instr,
    }


def _dataset_records(limit: int = 100, offset: int = 0) -> Dict:
    """Return a slice of dataset entries."""
    path = _dataset_path()
    if not path.exists():
        return {"records": [], "total": 0, "limit": limit, "offset": offset}

    max_limit = 500
    limit = min(max_limit, max(1, limit))
    offset = max(0, offset)

    records: List[Dict] = []
    total = 0
    with jsonlines.open(path, mode="r") as reader:
        for idx, row in enumerate(reader):
            total += 1
            if idx < offset:
                continue
            if len(records) >= limit:
                continue
            records.append(row)

    return {"records": records, "total": total, "limit": limit, "offset": offset}


def _process_file(task: TaskRecord, saved_path: Path) -> None:
    logger.info("Starting task %s for %s", task.id, task.filename)
    now = time.time()
    _update_task(
        task.id,
        status=TaskStatus.PROCESSING,
        message=f"Starting extraction (model: {task.model}, device: {task.device})",
        started_at=now,
        updated_at=now,
        progress=0.0,
        eta_seconds=None,
    )
    try:
        entries = process_pdf(
            saved_path,
            model=task.model,
            device=task.device,
            prompt=task.prompt,
            chunk_size=task.chunk_size,
            chunk_overlap=task.chunk_overlap,
            temperature=task.temperature,
            max_retries=task.max_retries,
            progress_callback=_progress_callback(task.id),
        )
        if not entries:
            raise RuntimeError("No entries generated from document")

        append_dataset_entries(entries)
        _update_task(
            task.id,
            status=TaskStatus.COMPLETED,
            generated=len(entries),
            progress=1.0,
            message="Completed",
            eta_seconds=0.0,
            updated_at=time.time(),
        )
        logger.info("Task %s completed with %s entries", task.id, len(entries))
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Task %s failed: %s", task.id, exc)
        _update_task(
            task.id,
            status=TaskStatus.FAILED,
            error=str(exc),
            message="Failed",
            progress=0.0,
        )


def _save_upload(file: UploadFile) -> Path:
    """Save an uploaded PDF to the raw directory."""
    raw_dir = Path(RAW_PDF_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename).name or "document.pdf"
    dest = raw_dir / original_name
    if dest.exists():
        dest = raw_dir / f"{dest.stem}_{uuid.uuid4().hex[:8]}{dest.suffix}"

    contents = file.file.read()
    dest.write_bytes(contents)
    return dest


@app.on_event("startup")
async def startup_event() -> None:
    ensure_processed_dir()
    try:
        test_ollama_connection()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Ollama connection check failed: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = web_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h3>UI not found. Upload endpoint: POST /upload</h3>")


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page() -> HTMLResponse:
    page_path = web_dir / "dataset.html"
    if page_path.exists():
        return HTMLResponse(page_path.read_text())
    return HTMLResponse("<h3>Dataset UI not found.</h3>")


@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    max_retries: Optional[int] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    selected_model = (model or "").strip() or OLLAMA_MODEL
    selected_device = (device or "").strip() or "auto"
    selected_prompt = (prompt or "").strip() or None
    selected_chunk_size = _to_int(chunk_size, "chunk_size")
    selected_chunk_overlap = _to_int(chunk_overlap, "chunk_overlap")
    selected_temperature = _to_float(temperature, "temperature")
    selected_max_retries = _to_int(max_retries, "max_retries")

    created_tasks: List[TaskRecord] = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        saved_path = _save_upload(file)
        task = _register_task(
            file.filename,
            model=selected_model,
            device=selected_device,
            prompt=selected_prompt,
            chunk_size=selected_chunk_size,
            chunk_overlap=selected_chunk_overlap,
            temperature=selected_temperature,
            max_retries=selected_max_retries,
        )
        created_tasks.append(task)
        background_tasks.add_task(executor.submit, _process_file, task, saved_path)

    return {"tasks": [asdict(task) for task in created_tasks]}


@app.get("/tasks")
async def list_tasks() -> Dict[str, List[Dict]]:
    with tasks_lock:
        return {"tasks": [asdict(task) for task in tasks.values()]}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict:
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return asdict(task)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dataset/summary")
async def dataset_summary() -> Dict:
    return _dataset_summary()


@app.get("/api/dataset/records")
async def dataset_records(limit: int = 100, offset: int = 0) -> Dict:
    return _dataset_records(limit=limit, offset=offset)


@app.get("/api/prompt")
async def get_prompt() -> Dict[str, str]:
    return {"prompt": INSTRUCTION_PROMPT}


@app.get("/api/config-defaults")
async def config_defaults() -> Dict:
    from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_RETRIES, TEMPERATURE

    return {
        "model": OLLAMA_MODEL,
        "device": "auto",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "temperature": TEMPERATURE,
        "max_retries": MAX_RETRIES,
        "prompt": INSTRUCTION_PROMPT,
    }
