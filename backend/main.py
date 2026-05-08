"""
main.py
-------
WHAT THIS FILE DOES:
  The HTTP entry point. Receives requests from the browser, orchestrates
  the pipeline (extract → agent → stream), and sends responses back.

TRIGGER CHAIN (the full picture):
  Browser sends POST /generate-prd
      └── FastAPI routes it to generate_prd()
              └── validate_pdf()             ← checks file is PDF and not too big
              └── extractor.extract_pdf_text()  ← returns List[str] chunks
              └── extractor.extraction_summary()  ← logs what was found
              └── agent.stream_prd()         ← returns async generator
              └── StreamingResponse(generator)   ← FastAPI streams it to browser

  Browser sends POST /export
      └── export_prd()
              └── exporter.to_markdown()     ← or to_docx() or to_pdf()
              └── FileResponse               ← sends file back for download

FASTAPI CONCEPTS USED:
  - @app.post()     = register a route (URL + HTTP method)
  - UploadFile      = FastAPI's type for multipart file uploads
  - Form(...)       = reads form fields from the request body
  - StreamingResponse = sends data chunk by chunk (doesn't buffer the whole response)
  - FileResponse    = sends a file from disk with correct Content-Type headers
  - HTTPException   = raises an HTTP error with a status code and message

CORS:
  The React frontend runs on localhost:3000.
  The FastAPI backend runs on localhost:8000.
  Browsers block cross-origin requests by default (CORS policy).
  CORSMiddleware tells FastAPI to allow requests from the frontend origin.
"""

import os
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from extractor import extract_pdf_text, extraction_summary
from agent import stream_prd
from exporter import to_markdown, to_docx, to_pdf

# ── Logging setup ─────────────────────────────────────────────────────────────
# Logs appear in your terminal when you run the server.
# Very useful for debugging — you'll see exactly what each request does.
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PRD Agent API",
    description="Upload a PDF → get a streaming Product Requirements Document",
    version="1.0.0",
)
app.state.max_upload_size = 10 * 1024 * 1024 
# ── CORS configuration ────────────────────────────────────────────────────────
# TRIGGER: Every browser request hits this middleware BEFORE reaching any route.
# It checks the Origin header and decides whether to allow the request.
#
# allow_origins: which frontends are allowed to call this API
# allow_methods: which HTTP methods are allowed
# allow_headers: which request headers are allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # React dev server
        "http://localhost:5173",    # Vite dev server (alternative)
        os.getenv("FRONTEND_URL", ""),  # production frontend URL from .env
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ── ROUTE 1: Health check ─────────────────────────────────────────────────────
# Triggered by: GET /health
# Purpose: lets you verify the server is running. Also useful for deployment
# platforms (Vercel, Railway, Render) that ping /health to check liveness.

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "prd-agent"}


# ── ROUTE 2: Main PRD generation ──────────────────────────────────────────────
# Triggered by: POST /generate-prd (from React frontend's api.js)
#
# FastAPI automatically:
#   - Parses multipart/form-data from the request body
#   - Fills `file` with the uploaded file object
#   - Fills `context` with the text field from the form
#
# Parameters:
#   file    = the PDF file (UploadFile = FastAPI's wrapper for uploaded files)
#   context = optional extra instructions from the user
#   model   = which LLM to use (defaults to Claude)

@app.post("/generate-prd")
async def generate_prd(
    file: UploadFile = File(...),           # ... means "required, no default"
    context: str = Form(default=""),        # optional form field
    model: str = Form(default="gemini-2.5-flash"),
):
    """
    THE ORCHESTRATOR — this function wires everything together.

    WHAT TRIGGERS WHAT, IN ORDER:
      1. validate_pdf(file)              → raises 400 if invalid
      2. file.read()                     → loads PDF bytes into memory
      3. extract_pdf_text(pdf_bytes)     → calls extractor.py, returns chunks
      4. extraction_summary(chunks)      → logs stats to terminal
      5. stream_prd(chunks, context)     → calls agent.py, returns async generator
      6. StreamingResponse(generator)    → FastAPI streams tokens to browser
         Content-Type: text/plain       → browser reads it as plain text stream

    WHY StreamingResponse?
      Without it, FastAPI would wait for the ENTIRE PRD to be generated (30-60s),
      then send it all at once. StreamingResponse sends each token the moment
      the LLM produces it — users see output in ~1 second.
    """
    logger.info(f"Received request: file={file.filename}, model={model}")

    # ── STEP 1: Validate the uploaded file ───────────────────────────────────
    await validate_pdf(file)

    # ── STEP 2: Read the PDF bytes into memory ───────────────────────────────
    # file.read() loads the entire file into a bytes object.
    # This is fine for PDFs up to ~20MB. For larger files, you'd stream the read.
    pdf_bytes = await file.read()
    logger.info(f"Read {len(pdf_bytes) / 1024:.1f} KB from {file.filename}")

    # ── STEP 3: Extract text chunks ──────────────────────────────────────────
    # This calls extractor.py → pdfplumber → clean → chunk
    # If the PDF is encrypted/image-only, extract_pdf_text() raises ValueError
    try:
        chunks = extract_pdf_text(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Log what was extracted (visible in your terminal)
    summary = extraction_summary(chunks)
    logger.info(f"Extraction complete: {summary}")

    # ── STEP 4: Create the streaming generator ───────────────────────────────
    # stream_prd() is an async generator function — calling it here does NOT
    # start the LLM call yet. It just creates the generator object.
    # The actual API call starts when StreamingResponse begins iterating it.
    generator = stream_prd(chunks, context, model)

    # ── STEP 5: Return a streaming response ──────────────────────────────────
    # StreamingResponse takes an async iterable and sends each yielded piece
    # to the browser immediately, keeping the connection open until the
    # generator is exhausted (i.e., the LLM finishes generating).
    return StreamingResponse(
        generator,
        media_type="text/plain",   # browser reads this as a text stream
        headers={
            # Tell the browser this is a streaming response, not a download
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )


# ── ROUTE 3: Export PRD ───────────────────────────────────────────────────────
# Triggered by: POST /export (from React after PRD is generated)
#
# The browser sends the finished Markdown text + desired format.
# We convert it and send back a file download.

@app.post("/export")
async def export_prd(
    content: str = Form(...),
    format: str = Form(default="markdown"),
    filename: str = Form(default="PRD"),
):
    logger.info(f"Export request: format={format}, filename={filename}")
    filename = Path(filename).stem  # strip any extension

    # Write to a persistent temp file (not inside a context manager)
    import tempfile
    suffix = ".md" if format == "markdown" else f".{format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    file_path = tmp.name

    if format == "markdown":
        to_markdown(content, file_path)
        media_type = "text/markdown"
    elif format == "docx":
        to_docx(content, file_path)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif format == "pdf":
        to_pdf(content, file_path)
        media_type = "application/pdf"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}")

    ext = "md" if format == "markdown" else format
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=f"{filename}.{ext}",
    )

# ── HELPER: PDF validation ────────────────────────────────────────────────────
# Triggered by: generate_prd() before anything else.
# Purpose: fail fast with a clear error rather than crashing deep in the pipeline.

async def validate_pdf(file: UploadFile) -> None:
    """
    Checks:
      1. File has a .pdf extension
      2. MIME type is application/pdf (what the browser reports)
      3. File size is under MAX_FILE_SIZE_MB

    RAISES: HTTPException(400) if any check fails.
    FastAPI automatically converts this to a JSON error response:
      {"detail": "error message here"}

    NOTE: These checks are not security-grade (a renamed .pdf can fool extension check).
    For production, also validate the PDF magic bytes: b"%PDF-" at file start.
    """
    # Check extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # Check MIME type (what the browser sends in Content-Type)
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected application/pdf."
        )

    # Check file size by reading headers (FastAPI provides this via file.size)
    # file.size is set when FastAPI parses the multipart form
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file.size / 1024 / 1024:.1f} MB. Maximum is {MAX_FILE_SIZE_MB} MB."
        )


# ── Entry point ───────────────────────────────────────────────────────────────
# When you run: python main.py
# This starts the Uvicorn ASGI server on port 8000.
# Uvicorn is an async web server — it handles concurrent requests efficiently.
# --reload flag in dev mode restarts the server when you save a file.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",          # "filename:FastAPI_instance_name"
        host="0.0.0.0",      # listen on all network interfaces
        port=8000,
        reload=True,         # auto-restart on file changes (dev only)
        log_level="info",
    )