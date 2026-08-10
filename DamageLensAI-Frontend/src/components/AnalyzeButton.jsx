import React from "react";
import { motion } from "framer-motion";

export default function AnalyzeButton({ onClick, loading, disabled }) {
  return (
    <motion.button
      whileHover={!disabled && !loading ? { scale: 1.02 } : {}}
      whileTap={!disabled && !loading ? { scale: 0.98 } : {}}
      onClick={onClick}
      disabled={disabled || loading}
      style={{
        ...s.btn,
        background: disabled
          ? "#3f3f46"
          : "linear-gradient(135deg, #00c6ff 0%, #0072ff 100%)",
        color: disabled ? "#71717a" : "#fff",
        boxShadow: disabled ? "none" : "0 4px 20px rgba(0,114,255,0.35)",
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      {loading ? "⏳ Processing..." : "🚀 Run AI Analysis"}
    </motion.button>
  );
}

const s = {
  btn: {
    width: "100%",
    padding: "15px",
    border: "none",
    borderRadius: 12,
    fontSize: "1rem",
    fontWeight: 700,
    fontFamily: "Inter, sans-serif",
    transition: "box-shadow 0.3s ease",
    letterSpacing: "0.3px",
  },
};
