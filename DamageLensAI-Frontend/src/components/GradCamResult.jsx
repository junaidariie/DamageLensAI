import React from "react";
import { getImageUrl } from "../api/client";

export default function GradCamResult({ gradcam, originalImage, mode }) {
  return (
    <div>
      <div style={s.infoBox}>
        Grad-CAM highlights which image regions drove the model's decision.
        <span style={{ color: "#ef4444" }}> Red</span> = high importance ·
        <span style={{ color: "#3b82f6" }}> Blue</span> = low importance.
      </div>
      <div style={s.grid} className="two-col-grid">
        <div style={s.imgCard}>
          <div style={s.label}>Original Image</div>
          <img src={getImageUrl(originalImage)} alt="original" style={s.img} />
        </div>
        <div style={s.imgCard}>
          <div style={s.label}>
            {mode === "fusion" ? "Fusion" : "ResNet"} Grad-CAM
          </div>
          <img src={getImageUrl(gradcam)} alt="gradcam" style={s.img} />
        </div>
      </div>
    </div>
  );
}

const s = {
  infoBox: {
    background: "rgba(0,198,255,0.07)",
    border: "1px solid rgba(0,198,255,0.2)",
    borderRadius: 10,
    padding: "11px 16px",
    fontSize: 13,
    color: "#a1a1aa",
    marginBottom: 16,
    lineHeight: 1.6,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 14,
  },
  imgCard: {
    background: "rgba(0,0,0,0.25)",
    border: "1px solid #27272a",
    borderRadius: 12,
    overflow: "hidden",
  },
  label: {
    fontSize: 12,
    fontWeight: 600,
    color: "#a1a1aa",
    textTransform: "uppercase",
    letterSpacing: "0.6px",
    padding: "9px 14px",
    borderBottom: "1px solid #27272a",
  },
  img: { width: "100%", display: "block", objectFit: "cover", maxHeight: 240 },
};
