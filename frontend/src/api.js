const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

/**
 * Streams a PRD from the backend.
 * @param {File} file - The PDF file to upload
 * @param {string} context - Optional extra instructions
 * @param {string} model - Gemini model name
 * @param {function} onChunk - Called with each text chunk as it streams in
 * @param {function} onDone - Called when streaming is complete
 * @param {function} onError - Called on error
 */
export async function generatePRD(file, context = "", model = "gemini-2.5-flash", onChunk, onDone, onError) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("context", context);
  formData.append("model", model);

  try {
    const response = await fetch(`${BASE_URL}/generate-prd`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      onChunk(chunk);
    }

    onDone();
  } catch (err) {
    onError(err.message);
  }
}

/**
 * Exports a PRD to a file and triggers browser download.
 * @param {string} content - The Markdown PRD content
 * @param {string} format - "markdown", "docx", or "pdf"
 * @param {string} filename - Base filename without extension
 */
export async function exportPRD(content, format = "markdown", filename = "PRD") {
  const formData = new FormData();
  formData.append("content", content);
  formData.append("format", format);
  formData.append("filename", filename);

  const response = await fetch(`${BASE_URL}/export`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Export failed" }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  // Trigger browser download
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const ext = format === "markdown" ? "md" : format;
  a.download = `${filename}.${ext}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}