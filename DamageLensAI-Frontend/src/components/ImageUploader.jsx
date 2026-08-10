import React, { useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";

export default function ImageUploader({ onFileSelect, selectedFile, scanning }) {
  const onDrop = useCallback(
    (accepted) => { if (accepted[0]) onFileSelect(accepted[0]); },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
  });

  const preview = selectedFile ? URL.createObjectURL(selectedFile) : null;

  return (
    <div>
      {!selectedFile ? (
        <div
          {...getRootProps()}
          style={{
            ...s.dropzone,
            borderColor: isDragActive ? "#00c6ff" : "#444",
            background: isDragActive ? "rgba(0,198,255,0.06)" : "rgba(255,255,255,0.02)",
          }}
        >
          <input {...getInputProps()} />
          <p style={{ fontSize: 36, marginBottom: 8 }}>📷</p>
          <p style={s.dropTitle}>
            {isDragActive ? "Drop it here!" : "Tap or Drag & Drop Vehicle Image"}
          </p>
          <p style={s.dropSub}>JPG · PNG · WEBP</p>
        </div>
      ) : (
        <div style={s.previewWrap}>
          <img src={preview} alt="preview" style={s.previewImg} />
          {/* Scan line overlay while analyzing */}
          {scanning && <div style={s.scanLine} />}
          {/* Loader overlay */}
          {scanning && (
            <div style={s.loaderOverlay}>
              <div style={s.spinner} />
              <p style={s.loaderTitle}>🧠 ANALYZING...</p>
              <p style={s.loaderSub} id="loaderStatus">Running AI pipeline...</p>
            </div>
          )}
          {/* Re-upload button */}
          {!scanning && (
            <div
              {...getRootProps()}
              style={s.reuploadBtn}
            >
              <input {...getInputProps()} />
              <span>📷 Change Image</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const s = {
  dropzone: {
    height: 180,
    border: "2px dashed",
    borderRadius: 16,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    transition: "all 0.3s ease",
  },
  dropTitle: { color: "#a1a1aa", fontWeight: 600, fontSize: 15, marginBottom: 4 },
  dropSub: { color: "#52525b", fontSize: 13 },
  previewWrap: {
    position: "relative",
    borderRadius: 16,
    overflow: "hidden",
    border: "1px solid #27272a",
    background: "#09090b",
    maxHeight: 360,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  previewImg: {
    width: "100%",
    maxHeight: 360,
    objectFit: "contain",
    display: "block",
  },
  scanLine: {
    position: "absolute",
    left: 0,
    width: "100%",
    height: 4,
    background: "#00c6ff",
    boxShadow: "0 0 14px #00c6ff, 0 0 28px #00c6ff",
    animation: "scanMove 2s ease-in-out infinite",
    zIndex: 5,
    filter: "blur(1px)",
  },
  loaderOverlay: {
    position: "absolute",
    inset: 0,
    background: "rgba(0,0,0,0.65)",
    backdropFilter: "blur(4px)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 10,
    gap: 10,
  },
  spinner: {
    width: 48,
    height: 48,
    border: "4px solid rgba(0,198,255,0.2)",
    borderTop: "4px solid #00c6ff",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  loaderTitle: { color: "#fff", fontWeight: 700, letterSpacing: 1, fontSize: 15 },
  loaderSub: { color: "#00c6ff", fontSize: 13 },
  reuploadBtn: {
    position: "absolute",
    bottom: 10,
    right: 10,
    background: "rgba(0,0,0,0.7)",
    border: "1px solid #3f3f46",
    borderRadius: 8,
    color: "#a1a1aa",
    fontSize: 12,
    padding: "6px 12px",
    cursor: "pointer",
    zIndex: 4,
  },
};
