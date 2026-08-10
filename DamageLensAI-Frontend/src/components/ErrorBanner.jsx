import React from "react";

export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div style={s.banner}>
      <span style={{ fontSize: 16 }}>⚠️</span>
      <span style={s.msg}>{message}</span>
      <button onClick={onDismiss} style={s.close}>✕</button>
    </div>
  );
}

const s = {
  banner: {
    display: "flex",
    alignItems: "flex-start",
    gap: 10,
    padding: "12px 16px",
    background: "rgba(239,68,68,0.08)",
    border: "1px solid rgba(239,68,68,0.3)",
    borderLeft: "4px solid #ef4444",
    borderRadius: 10,
  },
  msg: { flex: 1, fontSize: 13, color: "#fca5a5", lineHeight: 1.5 },
  close: {
    background: "none",
    border: "none",
    color: "#ef4444",
    cursor: "pointer",
    fontSize: 14,
    padding: 0,
    flexShrink: 0,
  },
};
