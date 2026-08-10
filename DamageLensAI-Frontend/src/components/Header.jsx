import React from "react";

export default function Header() {
  return (
    <div style={s.wrap}>
      <div style={s.shimmerText}>🚗 Car Damage AI</div>
      <p style={s.sub}>Fusion Intelligence: ResNet + YOLO + Grad-CAM</p>
      <div style={s.warning} className="warning-box">
        <span style={{ fontSize: 18 }}>⏱️</span>
        <span>
          <b>Note:</b> First analysis may take 3–4 min while models warm up.
          Subsequent requests are much faster.
        </span>
      </div>
    </div>
  );
}

const s = {
  wrap: { textAlign: "center", marginBottom: 32 },
  shimmerText: {
    fontSize: "clamp(1.8rem, 5vw, 2.6rem)",
    fontWeight: 800,
    background:
      "linear-gradient(90deg, #e2e8f0 0%, #ffffff 20%, #00c6ff 50%, #e2e8f0 75%, #e2e8f0 100%)",
    backgroundSize: "200% auto",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
    animation: "shimmer 4s linear infinite",
    marginBottom: 6,
  },
  sub: { color: "#a1a1aa", fontSize: "1rem", marginBottom: 20 },
  warning: {
    display: "flex",
    alignItems: "flex-start",
    gap: 10,
    background: "rgba(0,198,255,0.08)",
    border: "1px solid rgba(0,198,255,0.25)",
    borderLeft: "4px solid #00c6ff",
    color: "#e2e8f0",
    padding: "11px 18px",
    borderRadius: 10,
    fontSize: "0.88rem",
    maxWidth: 560,
    width: "100%",
    textAlign: "left",
    boxSizing: "border-box",
  },
};
