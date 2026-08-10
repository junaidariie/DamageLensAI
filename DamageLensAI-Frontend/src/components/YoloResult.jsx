import React from "react";
import { getImageUrl } from "../api/client";

export default function YoloResult({ yolo }) {
  if (!yolo) return null;
  const { image, detections, total_detections } = yolo;
  const hasDetections = total_detections > 0;

  return (
    <div style={s.grid} className="two-col-grid">
      {/* Left: annotated image */}
      <div style={s.imgCard}>
        <h3 style={s.cardTitle}>Bounding Boxes</h3>
        <img src={getImageUrl(image)} alt="YOLO output" style={s.img} />
      </div>

      {/* Right: detection log */}
      <div style={s.logBox}>
        <h3 style={s.cardTitle}>Detection Log</h3>

        {!hasDetections ? (
          <div style={s.noDetect}>🟢 No damage regions detected.</div>
        ) : (
          <>
            <div style={s.foundBanner}>
              🔴 Found {total_detections} damage region{total_detections > 1 ? "s" : ""}
            </div>
            {detections.map((d, i) => (
              <div key={i} style={s.detItem}>
                <div style={s.detTop}>
                  <span style={s.detRegion}>Region {i + 1}</span>
                  <span style={s.detConf}>
                    {d.confidence != null
                      ? `${(d.confidence * 100).toFixed(1)}%`
                      : "—"}
                  </span>
                </div>
                <div style={s.detLabel}>{d.label || d.class || "Damage"}</div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

const s = {
  grid: {
    display: "grid",
    gridTemplateColumns: "1.5fr 1fr",
    gap: 18,
  },
  imgCard: {
    background: "rgba(0,0,0,0.2)",
    border: "1px solid #27272a",
    borderRadius: 14,
    padding: "16px 18px",
  },
  logBox: {
    background: "rgba(0,0,0,0.2)",
    border: "1px solid #27272a",
    borderRadius: 14,
    padding: "16px 18px",
  },
  cardTitle: { fontSize: "1rem", fontWeight: 700, color: "#e2e8f0", marginBottom: 12 },
  img: { width: "100%", borderRadius: 10, display: "block" },
  noDetect: { color: "#a1a1aa", fontSize: 14, padding: "10px 0" },
  foundBanner: {
    color: "#fbbf24",
    fontWeight: 700,
    fontSize: 14,
    marginBottom: 12,
  },
  detItem: {
    background: "#27272a",
    borderLeft: "4px solid #00c6ff",
    borderRadius: 8,
    padding: "10px 14px",
    marginBottom: 10,
    boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
  },
  detTop: { display: "flex", justifyContent: "space-between", marginBottom: 3 },
  detRegion: { fontWeight: 700, color: "#e2e8f0", fontSize: 14 },
  detConf: { color: "#00c6ff", fontWeight: 700, fontSize: 14, fontFamily: "monospace" },
  detLabel: { color: "#a1a1aa", fontSize: 13 },
};
