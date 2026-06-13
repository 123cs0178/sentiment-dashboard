import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";
import {
  Zap, Trash2, Send, RotateCcw, ChevronRight, Activity,
  ThumbsUp, ThumbsDown, AlertCircle, CheckCircle2
} from "lucide-react";

const API = "https://sentiment-dashboard-cox0.onrender.com";

// ── Helpers ──────────────────────────────────────────────────────────────────
const scoreColor = (score) =>
  score >= 60 ? "var(--pos)" : score >= 40 ? "var(--warn)" : "var(--neg)";

const timeAgo = (ts) => {
  const d = Math.floor((Date.now() / 1000 - ts));
  if (d < 60) return `${d}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  return `${Math.floor(d / 3600)}h ago`;
};

const SAMPLE_REVIEWS = [
  "This product is absolutely fantastic! Best purchase I've made this year.",
  "Terrible quality, broke after two days. Complete waste of money.",
  "Decent product, does what it says. Nothing extraordinary but gets the job done.",
  "Fast shipping, great packaging. The item exceeded my expectations!",
  "Very disappointed. Customer service was unhelpful and the product didn't match the description.",
  "Works perfectly fine. Good value for the price point.",
];

// ── Sub-components ────────────────────────────────────────────────────────────
const ScoreMeter = ({ score }) => {
  const color = scoreColor(score);
  return (
    <div style={{ margin: "8px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: "var(--text-dim)", fontFamily: "var(--font-mono)", fontSize: 11 }}>SENTIMENT SCORE</span>
        <span style={{ color, fontFamily: "var(--font-mono)", fontWeight: 600 }}>{score.toFixed(1)}</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: "var(--border)", overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${score}%`, borderRadius: 3,
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          transition: "width 0.5s ease"
        }} />
      </div>
    </div>
  );
};

const Tag = ({ label, confidence }) => (
  <span style={{
    display: "inline-flex", alignItems: "center", gap: 4,
    padding: "2px 10px", borderRadius: 999,
    fontSize: 11, fontFamily: "var(--font-mono)", fontWeight: 600,
    textTransform: "uppercase", letterSpacing: "0.08em",
    background: label === "positive" ? "#22d3a520" : "#f45b7a20",
    color: label === "positive" ? "var(--pos)" : "var(--neg)",
    border: `1px solid ${label === "positive" ? "#22d3a540" : "#f45b7a40"}`,
  }}>
    {label === "positive" ? <ThumbsUp size={10} /> : <ThumbsDown size={10} />}
    {label} · {confidence.toFixed(1)}%
  </span>
);

const StatCard = ({ label, value, sub, color }) => (
  <div style={{
    background: "var(--surface)", border: "1px solid var(--border)",
    borderRadius: "var(--radius)", padding: "18px 20px",
    display: "flex", flexDirection: "column", gap: 4,
  }}>
    <span style={{ color: "var(--text-dim)", fontSize: 11, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</span>
    <span style={{ fontSize: 28, fontWeight: 600, fontFamily: "var(--font-mono)", color: color || "var(--text)" }}>{value}</span>
    {sub && <span style={{ color: "var(--text-dim)", fontSize: 12 }}>{sub}</span>}
  </div>
);

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, padding: "8px 12px", fontSize: 12, fontFamily: "var(--font-mono)" }}>
      <div style={{ color: "var(--text)" }}>{payload[0].name}</div>
      <div style={{ color: "var(--accent)", fontWeight: 600 }}>{payload[0].value}</div>
    </div>
  );
};

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("single");
  const [batchInput, setBatchInput] = useState(SAMPLE_REVIEWS.join("\n"));
  const [batchResults, setBatchResults] = useState([]);
  const textareaRef = useRef(null);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/history`);
      setHistory(await res.json());
    } catch {}
  }, []);

  const fetchAnalytics = useCallback(async () => {
    try {
      const res = await fetch(`${API}/analytics`);
      setAnalytics(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchAnalytics();
    const id = setInterval(() => { fetchHistory(); fetchAnalytics(); }, 10000);
    return () => clearInterval(id);
  }, [fetchHistory, fetchAnalytics]);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();
      setResult(data);
      await fetchHistory();
      await fetchAnalytics();
    } catch (e) {
      setError(e.message || "Request failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleBatch = async () => {
    const texts = batchInput.split("\n").map(t => t.trim()).filter(Boolean);
    if (!texts.length) return;
    setBatchLoading(true); setError(""); setBatchResults([]);
    try {
      const res = await fetch(`${API}/analyze/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts }),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      setBatchResults(await res.json());
      await fetchHistory();
      await fetchAnalytics();
    } catch (e) {
      setError(e.message || "Batch request failed.");
    } finally {
      setBatchLoading(false);
    }
  };

  const handleClear = async () => {
    await fetch(`${API}/history`, { method: "DELETE" });
    setHistory([]); setAnalytics(null); setResult(null); setBatchResults([]);
  };

  const distData = analytics?.distribution
    ? Object.entries(analytics.distribution).map(([k, v]) => ({ name: k, count: v }))
    : [];

  const pieData = analytics?.total
    ? [
      { name: "Positive", value: analytics.positive },
      { name: "Negative", value: analytics.negative },
    ]
    : [];

  const tabStyle = (t) => ({
    padding: "7px 16px", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 500,
    background: activeTab === t ? "var(--accent)" : "transparent",
    color: activeTab === t ? "#fff" : "var(--text-dim)",
    border: "none", transition: "all 0.15s",
  });

  return (
    <div style={{ minHeight: "100vh", padding: "28px 24px", maxWidth: 1280, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: 32, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <Activity size={22} color="var(--accent)" />
            <h1 style={{ fontSize: 22, fontWeight: 600, fontFamily: "var(--font-mono)", letterSpacing: "-0.02em" }}>
              Sentiment<span style={{ color: "var(--accent)" }}>Lab</span>
            </h1>
          </div>
          <p style={{ color: "var(--text-dim)", fontSize: 13 }}>DistilBERT · Amazon Reviews · Real-time NLP inference</p>
        </div>
        <button onClick={handleClear} style={{
          display: "flex", alignItems: "center", gap: 6, padding: "7px 14px",
          background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)",
          color: "var(--text-dim)", cursor: "pointer", fontSize: 13, transition: "border-color 0.15s",
        }}
          onMouseEnter={e => e.currentTarget.style.borderColor = "var(--neg)"}
          onMouseLeave={e => e.currentTarget.style.borderColor = "var(--border)"}
        >
          <Trash2 size={14} /> Clear history
        </button>
      </div>

      {/* Stats row */}
      {analytics?.total > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
          <StatCard label="Total Analyzed" value={analytics.total} />
          <StatCard label="Positive" value={`${analytics.positive_pct}%`} sub={`${analytics.positive} reviews`} color="var(--pos)" />
          <StatCard label="Avg Sentiment" value={`${analytics.avg_sentiment_score}`} sub="out of 100" color="var(--accent)" />
          <StatCard label="Avg Confidence" value={`${analytics.avg_confidence}%`} color="var(--warn)" />
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 20 }}>

        {/* Left column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Input panel */}
          <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", overflow: "hidden" }}>
            <div style={{ display: "flex", gap: 4, padding: "10px 14px", borderBottom: "1px solid var(--border)" }}>
              <button style={tabStyle("single")} onClick={() => setActiveTab("single")}>Single</button>
              <button style={tabStyle("batch")} onClick={() => setActiveTab("batch")}>Batch</button>
            </div>

            <div style={{ padding: 16 }}>
              {activeTab === "single" ? (
                <>
                  <textarea
                    ref={textareaRef}
                    value={text}
                    onChange={e => setText(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter" && e.metaKey) handleAnalyze(); }}
                    placeholder="Paste a product review here…"
                    rows={5}
                    style={{
                      width: "100%", background: "var(--bg)", border: "1px solid var(--border)",
                      borderRadius: 6, color: "var(--text)", fontFamily: "var(--font-sans)",
                      fontSize: 14, padding: "10px 12px", resize: "vertical", outline: "none",
                      lineHeight: 1.6,
                    }}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
                    <span style={{ color: "var(--text-dim)", fontSize: 12, fontFamily: "var(--font-mono)" }}>{text.length} chars · ⌘+Enter to run</span>
                    <button onClick={handleAnalyze} disabled={loading || !text.trim()} style={{
                      display: "flex", alignItems: "center", gap: 6, padding: "8px 18px",
                      background: "var(--accent)", border: "none", borderRadius: 6,
                      color: "#fff", fontWeight: 600, fontSize: 13, cursor: loading ? "not-allowed" : "pointer",
                      opacity: !text.trim() ? 0.4 : 1, transition: "opacity 0.15s",
                    }}>
                      {loading ? <RotateCcw size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Zap size={14} />}
                      {loading ? "Analyzing…" : "Analyze"}
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <p style={{ color: "var(--text-dim)", fontSize: 12, marginBottom: 8 }}>One review per line · max 20</p>
                  <textarea
                    value={batchInput}
                    onChange={e => setBatchInput(e.target.value)}
                    rows={8}
                    style={{
                      width: "100%", background: "var(--bg)", border: "1px solid var(--border)",
                      borderRadius: 6, color: "var(--text)", fontFamily: "var(--font-mono)",
                      fontSize: 12, padding: "10px 12px", resize: "vertical", outline: "none", lineHeight: 1.7,
                    }}
                  />
                  <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 10 }}>
                    <button onClick={handleBatch} disabled={batchLoading} style={{
                      display: "flex", alignItems: "center", gap: 6, padding: "8px 18px",
                      background: "var(--accent)", border: "none", borderRadius: 6,
                      color: "#fff", fontWeight: 600, fontSize: 13, cursor: batchLoading ? "not-allowed" : "pointer",
                    }}>
                      {batchLoading ? <RotateCcw size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Send size={14} />}
                      {batchLoading ? "Processing…" : "Run Batch"}
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "10px 14px", background: "#f45b7a15", border: "1px solid #f45b7a40", borderRadius: "var(--radius)", color: "var(--neg)", fontSize: 13 }}>
              <AlertCircle size={15} style={{ marginTop: 1, flexShrink: 0 }} /> {error}
            </div>
          )}

          {/* Single result */}
          {result && activeTab === "single" && (
            <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 18 }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 12, gap: 8 }}>
                <p style={{ color: "var(--text)", fontSize: 14, lineHeight: 1.5, maxWidth: "70%" }}>"{result.text}"</p>
                <Tag label={result.label} confidence={result.confidence} />
              </div>
              <ScoreMeter score={result.sentiment_score} />
            </div>
          )}

          {/* Batch results */}
          {batchResults.length > 0 && activeTab === "batch" && (
            <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", overflow: "hidden" }}>
              <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 8 }}>
                <CheckCircle2 size={14} color="var(--pos)" />
                <span style={{ fontSize: 13, fontFamily: "var(--font-mono)" }}>{batchResults.length} results</span>
              </div>
              {batchResults.map((r, i) => (
                <div key={i} style={{
                  padding: "12px 16px", borderBottom: i < batchResults.length - 1 ? "1px solid var(--border)" : "none",
                  display: "flex", gap: 12, alignItems: "center",
                }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-dim)", minWidth: 20 }}>{String(i + 1).padStart(2, "0")}</span>
                  <p style={{ flex: 1, fontSize: 13, color: "var(--text)", lineHeight: 1.4 }}>{r.text}</p>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4, minWidth: 120 }}>
                    <Tag label={r.label} confidence={r.confidence} />
                    <div style={{ width: 120, height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${r.sentiment_score}%`, background: scoreColor(r.sentiment_score), borderRadius: 2 }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Charts */}
          {analytics?.total > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 16 }}>
                <p style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>Score distribution</p>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={distData} barCategoryGap="30%">
                    <XAxis dataKey="name" tick={{ fill: "var(--text-dim)", fontSize: 10, fontFamily: "var(--font-mono)" }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "var(--text-dim)", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {distData.map((entry, i) => (
                        <Cell key={i} fill={["var(--neg)", "#f45b7a99", "var(--warn)", "#22d3a599", "var(--pos)"][i]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 16 }}>
                <p style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>Pos / Neg split</p>
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" cx="50%" cy="50%" outerRadius={60} innerRadius={35} paddingAngle={3}>
                      <Cell fill="var(--pos)" />
                      <Cell fill="var(--neg)" />
                    </Pie>
                    <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "var(--text-dim)" }} />
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Right column — History */}
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-dim)" }}>History</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-dim)" }}>{history.length} / 100</span>
          </div>
          <div style={{ flex: 1, overflowY: "auto", maxHeight: 700 }}>
            {history.length === 0 ? (
              <div style={{ padding: 32, textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>
                No analyses yet. Run something above.
              </div>
            ) : (
              history.map((item, i) => (
                <div key={i} style={{
                  padding: "12px 16px",
                  borderBottom: "1px solid var(--border)",
                  cursor: "pointer",
                  transition: "background 0.1s",
                }}
                  onMouseEnter={e => e.currentTarget.style.background = "var(--surface2)"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                  onClick={() => { setText(item.text); setActiveTab("single"); }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                    <Tag label={item.label} confidence={item.confidence} />
                    <span style={{ fontSize: 10, color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>{timeAgo(item.timestamp)}</span>
                  </div>
                  <p style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: 1.4, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                    {item.text}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
                    <div style={{ flex: 1, height: 3, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${item.sentiment_score}%`, background: scoreColor(item.sentiment_score) }} />
                    </div>
                    <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-dim)" }}>{item.sentiment_score.toFixed(0)}</span>
                    <ChevronRight size={10} color="var(--text-dim)" />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
