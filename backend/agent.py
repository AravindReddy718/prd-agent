"""
agent.py
--------
WHAT THIS FILE DOES:
  Takes text chunks from extractor.py → builds a prompt → calls Gemini API →
  streams back the PRD token by token.

TRIGGER CHAIN INSIDE THIS FILE:
  stream_prd(chunks, context)             ← called by main.py
      └── build_prompt(chunks, context)   ← assembles the full prompt string
              └── joins all chunks        ← combines into one document body
              └── appends user context    ← optional focus instructions
              └── returns user_message string
      └── genai.GenerativeModel(...)      ← creates model with system prompt baked in
      └── model.generate_content(stream=True)  ← opens streaming connection to Gemini
              └── yields text chunks one by one
      └── async generator                 ← main.py iterates this → sends to browser

HOW TO GET YOUR FREE GEMINI API KEY:
  1. Go to https://aistudio.google.com/apikey
  2. Sign in with Google
  3. Click "Create API key"
  4. Copy it into your .env file:  GEMINI_API_KEY=AIzaSy-your-key-here
  No credit card. No billing. 1 million tokens/day free.

TEMPERATURE = 0.3 (not 0):
  0   = fully deterministic, robotic, repetitive phrasing
  0.3 = consistent structure but natural, varied language — ideal for documents
  1.0 = creative, unpredictable — wrong for PRDs which need precision
"""

import os
from typing import AsyncGenerator, List

import google.generativeai as genai
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
# Reads GEMINI_API_KEY from your .env file into os.environ.
# Must run before genai.configure() — otherwise the key is not available yet.
load_dotenv()

# ── Configure Gemini globally ─────────────────────────────────────────────────
# This is a one-time setup call — all subsequent genai calls use this key.
# If GEMINI_API_KEY is missing from .env, os.getenv() returns None and
# the first generate_content() call will raise:
#   google.api_core.exceptions.PermissionDenied: 403 API key not valid
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# The agent's fixed identity and strict output format.
# Passed to GenerativeModel as system_instruction — Gemini treats this as
# a persistent rule set that applies to every message in the session.
#
# WHY SYSTEM PROMPT SEPARATE FROM USER MESSAGE?
#   system_instruction = who the model IS and how it must behave (never changes)
#   user message       = the actual document content for this specific request
#   Keeping them separate ensures consistent PRD structure across all PDFs.

SYSTEM_PROMPT = """You are a senior product manager at a top-tier technology company.
You have shipped 20+ products and written hundreds of PRDs.

Your job: analyze the provided document content and generate a comprehensive,
well-structured Product Requirements Document (PRD) in Markdown format.

STRICT OUTPUT FORMAT — follow this exact structure:
# Product Requirements Document

## 1. Executive Summary
3-5 sentences. What is this product/feature? Why does it exist?

## 2. Problem Statement
What problem exists? Who has it? What is the business/user impact?

## 3. Goals & Objectives
- Primary goals (measurable, time-bound where possible)
- Secondary goals

## 4. Target Users & Personas
Describe primary and secondary personas. Include context, pain points, behaviors.

## 5. Functional Requirements
Every feature and capability the system must have. Be specific and numbered.

## 6. Non-Functional Requirements
Performance, security, scalability, accessibility, compliance, reliability.

## 7. User Stories
Format: "As a [user type], I want [action] so that [benefit]."
Write at least 6. Cover happy paths and edge cases.

## 8. Acceptance Criteria
For each major feature, define exactly what "done" means. Use checkboxes.

## 9. Out of Scope
Explicitly list what will NOT be built in this version. This is critical.

## 10. Success Metrics
How will success be measured? List KPIs with target values where possible.

## 11. Timeline & Milestones
Suggest a phased delivery approach. Phase 1, Phase 2, Phase 3.

## 12. Open Questions & Risks
List unknowns, dependencies, assumptions that need validation, blockers.

RULES:
- Base every section on the ACTUAL content of the document provided.
- Never invent features or requirements not implied by the document.
- Be specific. Vague requirements are useless.
- Use bullet points generously. Write like a PM who has shipped real products.
- Every section must have substantive content — never leave a section empty.
"""


# ── STEP 1: Build the prompt ──────────────────────────────────────────────────
# Triggered by: stream_prd() — runs first, before any API call.
# Purpose: assemble all text chunks into one coherent user message.

def build_prompt(chunks: List[str], context: str = "") -> str:
    """
    INPUT:  chunks — list of text strings returned by extractor.py
            context — optional user instruction e.g. "focus on mobile features"
    OUTPUT: one complete user message string, ready to send to Gemini

    WHAT HAPPENS INSIDE:
      1. All chunks joined with "---" separators so Gemini sees a single document
         (the separator helps it understand where PDF sections begin and end)
      2. Optional user context appended as an extra instruction
      3. Generation instruction added at the END of the message
         → LLMs respond better when the instruction follows the content,
           not buried before 10,000 words of PDF text

    WHY SEND ALL CHUNKS AT ONCE?
      Gemini 1.5 Flash has a 1 million token context window — large enough
      to hold most entire PDFs. We send everything at once so Gemini can
      reason across the full document without missing any section.
      For PDFs over ~500 pages, a RAG approach would be needed instead.
    """
    # Join all chunks into one document body with visible section breaks
    document_body = "\n\n---\n\n".join(chunks)

    user_message = f"""Here is the document content to analyze:

========== DOCUMENT START ==========
{document_body}
========== DOCUMENT END ==========
"""

    # Append optional focus instruction from the user
    # e.g. "Focus on the payment flow" or "Target audience: enterprise B2B"
    if context and context.strip():
        user_message += f"\n\nAdditional context from the requester: {context.strip()}"

    # Instruction goes LAST — models follow end-of-prompt instructions more reliably
    user_message += "\n\nNow generate the complete PRD based on this document. Follow the exact structure defined in your instructions."

    return user_message


# ── STEP 2: Stream the PRD ────────────────────────────────────────────────────
# Triggered by: main.py's generate_prd() endpoint.
# This is the core function — calls Gemini and yields tokens back one by one.

async def stream_prd(
    chunks: List[str],
    context: str = "",
    model: str = "gemini-2.5-flash",
) -> AsyncGenerator[str, None]:
    """
    INPUT:  chunks   — text chunks from extractor.py
            context  — optional extra instruction from the user
            model    — Gemini model name (see options below)
    OUTPUT: async generator that yields text strings as Gemini produces them

    WHY AN ASYNC GENERATOR?
      FastAPI's StreamingResponse needs an async iterable to stream data.
      Using `yield` turns this function into a generator — it pauses at each
      yield, sends that piece to the browser, then resumes when ready.
      This is how streaming works in Python without blocking the server.

    WHAT TRIGGERS WHAT, IN ORDER:
      1. build_prompt()               → assembles user_message from all chunks
      2. genai.GenerativeModel()      → configures model + bakes in system prompt
                                        (no network call yet — just setup)
      3. model.generate_content()     → opens streaming connection to Gemini API
                                        network call starts here
      4. for chunk in response        → iterates chunks as Gemini generates them
      5. yield chunk.text             → sends each piece to main.py → browser
                                        user sees text appear word by word

    GEMINI MODEL OPTIONS (all free tier):
      "gemini-1.5-flash"    → fastest, lowest latency, great quality  ← default
      "gemini-1.5-pro"      → smarter, better reasoning, slower
      "gemini-2.0-flash"    → newest model, also fast and free

    NOTE ON SYNC vs ASYNC:
      The google-generativeai SDK streams synchronously under the hood.
      We wrap it in an async generator so FastAPI's StreamingResponse can
      iterate it without blocking the event loop. This works well for a
      learning/development project. For high-concurrency production use,
      run the sync iteration in a thread pool executor.
    """
    # Step 1: Assemble the full user message from PDF chunks
    user_message = build_prompt(chunks, context)

    # Step 2: Create the Gemini model instance.
    #
    # GenerativeModel() is pure local setup — no network call here.
    # It configures:
    #   model_name         → which Gemini model to use
    #   system_instruction → our PRD template + PM persona, applied to every call
    #   generation_config  → controls how the model generates text:
    #     temperature=0.3      consistent but not robotic output
    #     max_output_tokens    hard ceiling on response length
    #                          8192 is the max for flash — plenty for a full PRD
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )

    # Step 3: Call the Gemini API with stream=True.
    #
    # generate_content() with stream=True:
    #   → opens a persistent HTTP connection to Gemini
    #   → returns a streaming response object immediately (model still generating)
    #   → each iteration gives a GenerateContentResponse with a text chunk
    #
    # Common errors at this line:
    #   PermissionDenied 403 → GEMINI_API_KEY is wrong or missing in .env
    #   ResourceExhausted 429 → hit free tier rate limit (15 req/min) — wait and retry
    response = gemini_model.generate_content(
        user_message,
        stream=True,
    )

    # Step 4: Iterate the stream, yield each text piece as it arrives.
    #
    # chunk.text = the NEW text in this chunk (not cumulative)
    # We guard with `if chunk.text` because some chunks carry only metadata
    # (finish_reason, safety ratings, usage stats) with empty text — skip those.
    for chunk in response:
        if chunk.text:
            yield chunk.text  # ← immediately forwarded to main.py → browser


# ── UTILITY: Local Ollama fallback (no internet, no API key needed) ───────────
# Use this if you want to run the agent completely offline on your own machine.
# Ollama runs open-source models (LLaMA 3, Mistral, Phi-3) locally.
#
# TO SWITCH TO OLLAMA:
#   1. Install Ollama: https://ollama.com
#   2. In terminal: ollama pull llama3
#   3. In terminal: ollama serve
#   4. In main.py: replace stream_prd with stream_prd_ollama
#
# HARDWARE NEEDED: 8GB RAM minimum, 16GB recommended for good output quality.

async def stream_prd_ollama(
    chunks: List[str],
    context: str = "",
    model: str = "llama3",
) -> AsyncGenerator[str, None]:
    """
    Same interface as stream_prd() but calls local Ollama instead of Gemini.
    Ollama exposes an OpenAI-compatible REST API at http://localhost:11434/v1.
    We use the openai Python library pointed at that local endpoint.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai  (needed for Ollama compatibility)")

    user_message = build_prompt(chunks, context)

    # Point the OpenAI client at your local Ollama server.
    # Ollama ignores the api_key value but the field is required by the SDK.
    ollama_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    response = ollama_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
        temperature=0.3,
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content