import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ClassificationResult from "./ClassificationResult";
import YoloResult from "./YoloResult";
import GradCamResult from "./GradCamResult";

const TABS = [
  { id: "pred", label: "📊 Prediction" },
  { id: "attention", label: "👀 Attention Maps" },
  { id: "yolo", label: "🎯 Localization" },
];

export default function ResultTabs({ result }) {
  const [active, setActive] = useState("pred");

  return (
    <div style={s.wrap}>
      {/* Tab bar */}
      <div style={s.tabBar}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            style={{
              ...s.tab,
              color: active === t.id ? "#00c6ff" : "#a1a1aa",
              background: active === t.id ? "rgba(0,198,255,0.1)" : "transparent",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          style={s.content}
          className="tab-content"
        >
          {active === "pred" && (
            <ClassificationResult
              classification={result.classification}
              mode={result.mode}
            />
          )}
          {active === "attention" && (
            <GradCamResult
              gradcam={result.gradcam}
              originalImage={result.original_image}
              mode={result.mode}
            />
          )}
          {active === "yolo" && <YoloResult yolo={result.yolo} />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

const s = {
  wrap: {
    background: "#18181b",
    border: "1px solid #27272a",
    borderRadius: 20,
    overflow: "hidden",
    animation: "slideUp 0.5s ease-out",
  },
  tabBar: {
    display: "flex",
    gap: 6,
    padding: "14px 16px 0",
    borderBottom: "1px solid #27272a",
    background: "#18181b",
    overflowX: "auto",
  },
  tab: {
    padding: "9px 18px",
    border: "none",
    borderRadius: "8px 8px 0 0",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600,
    fontFamily: "Inter, sans-serif",
    transition: "all 0.2s ease",
    whiteSpace: "nowrap",
    marginBottom: -1,
  },
  content: { padding: 22 },
};
