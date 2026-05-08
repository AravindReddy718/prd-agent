import { useState, useRef, useEffect } from "react";
import { generatePRD, exportPRD } from "./api";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [context, setContext] = useState("");
  const [prd, setPrd] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | streaming | done | error
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const outputRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll as PRD streams in
  useEffect(() => {
    if (outputRef.current && status === "streaming") {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [prd, status]);

  const handleFile = (f) => {
    if (f && f.type === "application/pdf") {
      setFile(f);
      setError("");
    } else {
      setError("Please upload a PDF file.");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  };

  const handleGenerate = () => {
    if (!file) return setError("Please upload a PDF first.");
    setPrd("");
    setError("");
    setStatus("loading");

    generatePRD(
      file,
      context,
      "gemini-2.5-flash",
      (chunk) => {
        setStatus("streaming");
        setPrd((prev) => prev + chunk);
      },
      () => setStatus("done"),
      (err) => {
        setError(err);
        setStatus("error");
      }
    );
  };

  const handleExport = async (format) => {
    setExporting(format);
    try {
      await exportPRD(prd, format, file ? file.name.replace(".pdf", "") : "PRD");
    } catch (err) {
      setError("Export failed: " + err.message);
    } finally {
      setExporting("");
    }
  };

  const reset = () => {
    setFile(null);
    setPrd("");
    setStatus("idle");
    setError("");
    setContext("");
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">PRD<span className="logo-accent">agent</span></span>
          </div>
          <p className="tagline">Drop a PDF. Get a PRD.</p>
        </div>
      </header>

      <main className="main">
        {/* Left Panel — Input */}
        <section className="panel panel-input">
          <div className="panel-label">01 / INPUT</div>

          {/* Drop Zone */}
          <div
            className={`dropzone ${dragOver ? "dragover" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {file ? (
              <div className="file-info">
                <span className="file-icon">📄</span>
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <div className="drop-prompt">
                <span className="drop-icon">↓</span>
                <span className="drop-text">Drop PDF here</span>
                <span className="drop-sub">or click to browse</span>
              </div>
            )}
          </div>

          {/* Context */}
          <div className="field">
            <label className="field-label">Focus instructions <span className="optional">(optional)</span></label>
            <textarea
              className="field-textarea"
              placeholder="e.g. Focus on mobile features, target enterprise B2B users..."
              value={context}
              onChange={(e) => setContext(e.target.value)}
              rows={3}
            />
          </div>

          {/* Error */}
          {error && <div className="error-msg">{error}</div>}

          {/* Actions */}
          <div className="actions">
            <button
              className={`btn-generate ${status === "loading" || status === "streaming" ? "loading" : ""}`}
              onClick={handleGenerate}
              disabled={status === "loading" || status === "streaming"}
            >
              {status === "loading" ? (
                <><span className="spinner" /> Extracting PDF…</>
              ) : status === "streaming" ? (
                <><span className="spinner" /> Generating PRD…</>
              ) : (
                "Generate PRD →"
              )}
            </button>

            {(status === "done" || status === "error") && (
              <button className="btn-reset" onClick={reset}>Start over</button>
            )}
          </div>

          {/* Export Buttons */}
          {status === "done" && (
            <div className="export-section">
              <div className="export-label">Export as</div>
              <div className="export-buttons">
                {["markdown", "docx", "pdf"].map((fmt) => (
                  <button
                    key={fmt}
                    className={`btn-export ${exporting === fmt ? "exporting" : ""}`}
                    onClick={() => handleExport(fmt)}
                    disabled={!!exporting}
                  >
                    {exporting === fmt ? "…" : fmt.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Right Panel — Output */}
        <section className="panel panel-output">
          <div className="panel-label">
            02 / OUTPUT
            {status === "streaming" && <span className="live-badge">LIVE</span>}
            {status === "done" && <span className="done-badge">DONE</span>}
          </div>

          <div className="output-area" ref={outputRef}>
            {status === "idle" && (
              <div className="output-placeholder">
                <span className="placeholder-icon">⬡</span>
                <p>Your PRD will appear here as it streams in, section by section.</p>
              </div>
            )}
            {status === "loading" && (
              <div className="output-placeholder">
                <span className="placeholder-icon pulse">⬡</span>
                <p>Reading your PDF…</p>
              </div>
            )}
            {(status === "streaming" || status === "done") && (
              <pre className="output-text">{prd}</pre>
            )}
            {status === "error" && (
              <div className="output-placeholder error">
                <p>Something went wrong. Check the error above and try again.</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}