import React from "react";

export default function ModelSelector({ selected, onChange }) {
  return (
    <div style={s.card}>
      <h3 style={s.title}>⚙️ Analysis Settings</h3>
      <p style={s.desc}>Select the neural network pipeline.</p>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        style={s.select}
      >
        <option value="fusion">🔬 Fusion — EfficientNet + ConvNeXt (84%)</option>
        <option value="resnet">⚡ ResNet-18 — Lightweight (77%)</option>
      </select>
    </div>
  );
}

const s = {
  card: {
    background: "rgba(0,0,0,0.25)",
    border: "1px solid #27272a",
    borderRadius: 16,
    padding: "18px 20px",
  },
  title: { fontSize: "1rem", fontWeight: 700, marginBottom: 4, color: "#e2e8f0" },
  desc: { fontSize: "0.85rem", color: "#a1a1aa", marginBottom: 10 },
  select: {
    width: "100%",
    background: "#27272a",
    border: "1px solid #3f3f46",
    padding: "13px 14px",
    borderRadius: 12,
    color: "#e2e8f0",
    outline: "none",
    fontSize: "0.95rem",
    cursor: "pointer",
    fontFamily: "Inter, sans-serif",
  },
};
