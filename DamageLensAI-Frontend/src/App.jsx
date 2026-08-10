import React, { useState, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Header from "./components/Header";
import ImageUploader from "./components/ImageUploader";
import ModelSelector from "./components/ModelSelector";
import AnalyzeButton from "./components/AnalyzeButton";
import ResultTabs from "./components/ResultTabs";
import ErrorBanner from "./components/ErrorBanner";
import { predictComprehensive } from "./api/client";

const STATUS_STEPS = [
  "Extracting image features...",
  "Running classification model...",
  "Generating Grad-CAM heatmap...",
  "Running YOLO detection...",
  "Finalizing results...",
];

export default function App() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("fusion");
  const [loading, setLoading] = useState(false);
  const [statusIdx, setStatusIdx] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const statusTimer = useRef(null);

  // Cycle through status messages while loading
  useEffect(() => {
    if (loading) {
      setStatusIdx(0);
      statusTimer.current = setInterval(() => {
        setStatusIdx((i) => (i + 1) % STATUS_STEPS.length);
      }, 4000);
    } else {
      clearInterval(statusTimer.current);
    }
    return () => clearInterval(statusTimer.current);
  }, [loading]);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictComprehensive(file, model);
      setResult(res.data);
    } catch (err) {
      if (err.code === "ECONNABORTED") {
        setError("Request timed out. The model may still be warming up — please try again.");
      } else if (err.response) {
        setError(`Server error ${err.response.status}: ${err.response.data?.detail || err.response.statusText}`);
      } else {
        setError(
          "Cannot reach the API. Make sure the backend is running and REACT_APP_API_URL is set correctly."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div style={s.page} className="app-page">
      {/* Radial glow background */}
      <div style={s.bgGlow} />

      <div style={s.container} className="app-container">
        <Header />

        {/* Controls row */}
        <div style={s.controlsGrid} className="controls-grid-responsive">
          <ImageUploader
            onFileSelect={(f) => { setFile(f); setResult(null); setError(null); }}
            selectedFile={file}
            scanning={loading}
          />
          <ModelSelector selected={model} onChange={setModel} />
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{ marginBottom: 16 }}
            >
              <ErrorBanner message={error} onDismiss={() => setError(null)} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Analyze button */}
        <AnalyzeButton onClick={handleAnalyze} loading={loading} disabled={!file} />

        {/* Loading status text below button */}
        <AnimatePresence>
          {loading && (
            <motion.p
              key={statusIdx}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={s.statusText}
            >
              {STATUS_STEPS[statusIdx]}
            </motion.p>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              style={s.resultsWrap}
            >
              <ResultTabs result={result} />
              <button onClick={handleReset} style={s.resetBtn}>
                🔄 New Analysis
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty state */}
        {!result && !loading && !error && (
          <div style={s.emptyState}>
            <p style={s.emptyIcon}>🚗</p>
            <p style={s.emptyTitle}>Upload a vehicle image to begin</p>
            <p style={s.emptySub}>
              The AI will classify damage, localize regions with YOLO, and explain predictions with Grad-CAM.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

const s = {
  page: {
    minHeight: "100vh",
    background: "#09090b",
    display: "flex",
    justifyContent: "center",
    alignItems: "flex-start",
    padding: "40px 20px 60px",
    position: "relative",
  },
  bgGlow: {
    position: "fixed",
    top: 0,
    right: 0,
    width: "50vw",
    height: "50vh",
    background: "radial-gradient(circle at top right, rgba(0,198,255,0.05) 0%, transparent 60%)",
    pointerEvents: "none",
    zIndex: 0,
  },
  container: {
    width: "100%",
    maxWidth: 860,
    background: "#18181b",
    borderRadius: 20,
    padding: "35px 32px",
    boxShadow: "0 20px 60px rgba(0,0,0,0.7)",
    border: "1px solid #27272a",
    animation: "slideUp 0.6s ease-out",
    position: "relative",
    zIndex: 1,
  },
  controlsGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 20,
    marginBottom: 22,
  },
  statusText: {
    textAlign: "center",
    color: "#00c6ff",
    fontSize: 13,
    marginTop: 10,
    animation: "pulse 2s ease-in-out infinite",
  },
  resultsWrap: { marginTop: 30 },
  resetBtn: {
    marginTop: 18,
    width: "100%",
    padding: "11px",
    background: "transparent",
    border: "1px solid #3f3f46",
    borderRadius: 10,
    color: "#a1a1aa",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "Inter, sans-serif",
    transition: "border-color 0.2s",
  },
  emptyState: {
    textAlign: "center",
    padding: "40px 20px",
    marginTop: 10,
  },
  emptyIcon: { fontSize: 48, marginBottom: 12, opacity: 0.35 },
  emptyTitle: { fontSize: 16, fontWeight: 600, color: "#52525b", marginBottom: 8 },
  emptySub: { fontSize: 13, color: "#3f3f46", maxWidth: 420, margin: "0 auto", lineHeight: 1.6 },
};
