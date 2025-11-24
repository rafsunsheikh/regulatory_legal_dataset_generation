"""FastAPI server exposing a web UI and upload API for the PDF pipeline."""

import logging
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure local imports work when running `uvicorn server:app`
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_PDF_DIR
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


tasks: Dict[str, TaskRecord] = {}
tasks_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=3)


def _update_task(task_id: str, **kwargs) -> None:
    with tasks_lock:
        if task_id not in tasks:
            return
        for key, value in kwargs.items():
            setattr(tasks[task_id], key, value)


def _register_task(filename: str, output_path: Optional[str] = None) -> TaskRecord:
    task_id = uuid.uuid4().hex
    record = TaskRecord(id=task_id, filename=filename, output_path=output_path)
    with tasks_lock:
        tasks[task_id] = record
    return record


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
            _update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=progress,
                eta_seconds=eta,
                message=f"Generating chunk {idx + 1}/{total}",
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


def _process_file(task: TaskRecord, saved_path: Path) -> None:
    logger.info("Starting task %s for %s", task.id, task.filename)
    now = time.time()
    _update_task(
        task.id,
        status=TaskStatus.PROCESSING,
        message="Starting extraction",
        started_at=now,
        updated_at=now,
        progress=0.0,
        eta_seconds=None,
    )
    try:
        entries = process_pdf(saved_path, progress_callback=_progress_callback(task.id))
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


@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    created_tasks: List[TaskRecord] = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        saved_path = _save_upload(file)
        task = _register_task(file.filename)
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
