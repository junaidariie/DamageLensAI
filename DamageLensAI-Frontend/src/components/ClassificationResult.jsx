import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, CartesianGrid,
} from "recharts";

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={s.tooltip}>
      <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{payload[0].payload.name}</span>
      <span style={{ color: "#00c6ff", fontWeight: 700 }}>
        {(payload[0].value * 100).toFixed(2)}%
      </span>
    </div>
  );
};

export default function ClassificationResult({ classification, mode }) {
  const [barWidth, setBarWidth] = useState(0);

  const entries = Object.entries(classification).sort((a, b) => b[1] - a[1]);
  const [topLabel, topConf] = entries[0];
  const chartData = entries.map(([name, value]) => ({ name, value }));
  const confPct = (topConf * 100).toFixed(2);

  useEffect(() => {
    const t = setTimeout(() => setBarWidth(parseFloat(confPct)), 120);
    return () => clearTimeout(t);
  }, [confPct]);

  return (
    <div style={s.wrap}>
      {/* Big prediction */}
      <div style={s.card}>
        <div style={s.bigText}>{topLabel}</div>
        <div style={s.confLabel}>
          Confidence Score: <span style={{ color: "#00c6ff" }}>{confPct}%</span>
          <span style={s.modelBadge}>{mode.toUpperCase()}</span>
        </div>

        {/* Animated progress bar */}
        <div style={s.progressTrack}>
          <div
            style={{
              ...s.progressFill,
              width: `${barWidth}%`,
            }}
          />
        </div>

        <h3 style={s.chartTitle}>Probability Distribution</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: -16, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="name"
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              angle={-30}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              domain={[0, 1]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
            <Bar dataKey="value" radius={[5, 5, 0, 0]}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.name === topLabel ? "#00c6ff" : "#0072ff"}
                  opacity={entry.name === topLabel ? 1 : 0.4}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const s = {
  wrap: {},
  card: {
    background: "rgba(0,0,0,0.2)",
    border: "1px solid #27272a",
    borderRadius: 14,
    padding: "20px 22px",
  },
  bigText: {
    fontSize: "clamp(1.6rem, 4vw, 2.4rem)",
    fontWeight: 800,
    background: "linear-gradient(45deg, #00c6ff, #0072ff)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
    marginBottom: 6,
  },
  confLabel: {
    fontWeight: 600,
    fontSize: "0.95rem",
    color: "#e2e8f0",
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 12,
  },
  modelBadge: {
    fontSize: 11,
    background: "rgba(0,198,255,0.12)",
    border: "1px solid rgba(0,198,255,0.3)",
    color: "#00c6ff",
    borderRadius: 5,
    padding: "2px 8px",
    fontWeight: 700,
    letterSpacing: "0.5px",
  },
  progressTrack: {
    background: "#27272a",
    borderRadius: 20,
    height: 12,
    overflow: "hidden",
    marginBottom: 22,
    boxShadow: "inset 0 2px 4px rgba(0,0,0,0.5)",
  },
  progressFill: {
    height: "100%",
    background: "linear-gradient(90deg, #00c6ff, #0072ff)",
    borderRadius: 20,
    transition: "width 1.4s cubic-bezier(0.22,1,0.36,1)",
  },
  chartTitle: {
    fontSize: "1rem",
    fontWeight: 600,
    color: "#e2e8f0",
    marginBottom: 10,
  },
  tooltip: {
    background: "#18181b",
    border: "1px solid #27272a",
    borderRadius: 8,
    padding: "8px 14px",
    display: "flex",
    flexDirection: "column",
    gap: 3,
    fontSize: 13,
  },
};
