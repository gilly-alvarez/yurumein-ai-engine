"use client";

import { useState, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type GapStatus =
  | "needs_review"
  | "research_needed"
  | "verified"
  | "resolved"
  | "dismissed";

type FlagReason =
  | "no_retrieved_docs"
  | "low_evidence"
  | "citation_rejected"
  | "gap_acknowledged"
  | "recurring_question";

interface KnowledgeGap {
  id: number;
  topic: string;
  normalized_question: string;
  frequency: number;
  flag_reason: FlagReason;
  first_seen: string;
  last_seen: string;
  status: GapStatus;
  admin_notes: string | null;
  source_reference: string | null;
}

interface GapListResponse {
  total: number;
  items: KnowledgeGap[];
}

interface ReportData {
  generated_at: string;
  period_days: number;
  new_this_week: number;
  status_breakdown: Record<string, number>;
  flag_reason_breakdown: Record<string, number>;
  top_topics: { topic: string; frequency: number }[];
  top_unresolved_gaps: KnowledgeGap[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// All calls go through the server-side Next.js proxy (/api/admin/...)
// The AI engine admin key (AI_ADMIN_KEY) stays server-side and is never sent to the browser.
const ADMIN_API = "/api/admin/knowledge-gaps";

const STATUS_LABELS: Record<GapStatus, string> = {
  needs_review: "Needs Review",
  research_needed: "Research Needed",
  verified: "Verified",
  resolved: "Resolved",
  dismissed: "Dismissed",
};

const STATUS_COLORS: Record<GapStatus, string> = {
  needs_review: "bg-yellow-100 text-yellow-800",
  research_needed: "bg-orange-100 text-orange-800",
  verified: "bg-blue-100 text-blue-800",
  resolved: "bg-green-100 text-green-800",
  dismissed: "bg-gray-100 text-gray-600",
};

const REASON_LABELS: Record<FlagReason, string> = {
  no_retrieved_docs: "No Sources Found",
  low_evidence: "Thin Evidence",
  citation_rejected: "Citation Rejected",
  gap_acknowledged: "AI Acknowledged Gap",
  recurring_question: "Recurring Question",
};

const REASON_COLORS: Record<FlagReason, string> = {
  no_retrieved_docs: "bg-red-100 text-red-700",
  low_evidence: "bg-orange-100 text-orange-700",
  citation_rejected: "bg-purple-100 text-purple-700",
  gap_acknowledged: "bg-blue-100 text-blue-700",
  recurring_question: "bg-pink-100 text-pink-700",
};

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function KnowledgeGapsPage() {
  const [view, setView] = useState<"list" | "report">("list");
  const [gaps, setGaps] = useState<KnowledgeGap[]>([]);
  const [total, setTotal] = useState(0);
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [reasonFilter, setReasonFilter] = useState<string>("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;

  // Modal state
  const [selected, setSelected] = useState<KnowledgeGap | null>(null);
  const [editStatus, setEditStatus] = useState<string>("");
  const [editNotes, setEditNotes] = useState<string>("");
  const [editRef, setEditRef] = useState<string>("");
  const [saving, setSaving] = useState(false);

  const fetchGaps = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(page * PAGE_SIZE),
      });
      if (statusFilter) params.set("status", statusFilter);
      if (reasonFilter) params.set("flag_reason", reasonFilter);
      const res = await fetch(`${ADMIN_API}?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: GapListResponse = await res.json();
      setGaps(data.items);
      setTotal(data.total);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load gaps");
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter, reasonFilter]);

  const fetchReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${ADMIN_API}/report`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setReport(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load report");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (view === "list") fetchGaps();
    else fetchReport();
  }, [view, fetchGaps, fetchReport]);

  const openModal = (gap: KnowledgeGap) => {
    setSelected(gap);
    setEditStatus(gap.status);
    setEditNotes(gap.admin_notes ?? "");
    setEditRef(gap.source_reference ?? "");
  };

  const saveModal = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      const res = await fetch(`${ADMIN_API}/${selected.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          status: editStatus || null,
          admin_notes: editNotes || null,
          source_reference: editRef || null,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSelected(null);
      fetchGaps();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const deleteGap = async (id: number) => {
    if (!confirm("Delete this knowledge gap record? This cannot be undone.")) return;
    try {
      const res = await fetch(`${ADMIN_API}/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      fetchGaps();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "Delete failed");
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Knowledge Gap Review</h1>
        <p className="mt-1 text-sm text-gray-500">
          Review topics where the AI had insufficient evidence, missing sources, or repeated questions.
          No entry here modifies verified content — admin review only.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-200">
        <button
          onClick={() => setView("list")}
          className={`pb-2 px-4 text-sm font-medium border-b-2 transition-colors ${
            view === "list"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
        >
          Review Queue
        </button>
        <button
          onClick={() => setView("report")}
          className={`pb-2 px-4 text-sm font-medium border-b-2 transition-colors ${
            view === "report"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
        >
          Weekly Report
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* LIST VIEW */}
      {view === "list" && (
        <>
          {/* Filters */}
          <div className="flex flex-wrap gap-3 mb-4">
            <select
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
              className="border border-gray-300 rounded px-3 py-1.5 text-sm text-gray-700"
            >
              <option value="">All Statuses</option>
              {Object.entries(STATUS_LABELS).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
            <select
              value={reasonFilter}
              onChange={(e) => { setReasonFilter(e.target.value); setPage(0); }}
              className="border border-gray-300 rounded px-3 py-1.5 text-sm text-gray-700"
            >
              <option value="">All Signals</option>
              {Object.entries(REASON_LABELS).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
            <button
              onClick={fetchGaps}
              className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded text-sm text-gray-700"
            >
              Refresh
            </button>
            <span className="ml-auto text-sm text-gray-500 self-center">
              {total} gap{total !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Table */}
          {loading ? (
            <div className="text-center py-12 text-gray-400">Loading…</div>
          ) : gaps.length === 0 ? (
            <div className="text-center py-12 text-gray-400">No knowledge gaps found.</div>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-gray-200">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                  <tr>
                    <th className="px-4 py-3 text-left">Topic</th>
                    <th className="px-4 py-3 text-left">Question (normalized)</th>
                    <th className="px-4 py-3 text-left">Signal</th>
                    <th className="px-4 py-3 text-left">Status</th>
                    <th className="px-4 py-3 text-right">Freq.</th>
                    <th className="px-4 py-3 text-left">Last Seen</th>
                    <th className="px-4 py-3 text-left">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {gaps.map((g) => (
                    <tr key={g.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-medium text-gray-900 whitespace-nowrap">
                        {g.topic}
                      </td>
                      <td className="px-4 py-3 text-gray-600 max-w-xs truncate" title={g.normalized_question}>
                        {g.normalized_question}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${REASON_COLORS[g.flag_reason]}`}>
                          {REASON_LABELS[g.flag_reason]}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_COLORS[g.status]}`}>
                          {STATUS_LABELS[g.status]}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-gray-900 font-semibold">
                        {g.frequency}
                      </td>
                      <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                        {new Date(g.last_seen).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <button
                          onClick={() => openModal(g)}
                          className="text-blue-600 hover:underline mr-3 text-xs"
                        >
                          Review
                        </button>
                        <button
                          onClick={() => deleteGap(g.id)}
                          className="text-red-500 hover:underline text-xs"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="flex justify-center gap-2 mt-4">
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
                className="px-3 py-1 text-sm border rounded disabled:opacity-40"
              >
                Previous
              </button>
              <span className="text-sm text-gray-600 self-center">
                Page {page + 1} of {Math.ceil(total / PAGE_SIZE)}
              </span>
              <button
                disabled={(page + 1) * PAGE_SIZE >= total}
                onClick={() => setPage((p) => p + 1)}
                className="px-3 py-1 text-sm border rounded disabled:opacity-40"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}

      {/* REPORT VIEW */}
      {view === "report" && (
        <>
          {loading ? (
            <div className="text-center py-12 text-gray-400">Generating report…</div>
          ) : report ? (
            <div className="space-y-6">
              {/* Summary cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <SummaryCard
                  label="New This Week"
                  value={report.new_this_week}
                  color="blue"
                />
                {Object.entries(report.status_breakdown).map(([k, v]) => (
                  <SummaryCard
                    key={k}
                    label={STATUS_LABELS[k as GapStatus] ?? k}
                    value={v}
                    color="gray"
                  />
                ))}
              </div>

              {/* Top topics */}
              <div>
                <h2 className="text-base font-semibold text-gray-800 mb-3">Top Knowledge Topics</h2>
                <div className="space-y-2">
                  {report.top_topics.map(({ topic, frequency }) => (
                    <div key={topic} className="flex items-center gap-3">
                      <span className="w-48 text-sm text-gray-700 truncate">{topic}</span>
                      <div className="flex-1 bg-gray-100 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{
                            width: `${Math.min(100, (frequency / (report.top_topics[0]?.frequency || 1)) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-sm text-gray-600 w-8 text-right">{frequency}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Signal breakdown */}
              <div>
                <h2 className="text-base font-semibold text-gray-800 mb-3">Signal Breakdown</h2>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(report.flag_reason_breakdown).map(([k, v]) => (
                    <span key={k} className={`px-3 py-1 rounded-full text-sm ${REASON_COLORS[k as FlagReason] ?? "bg-gray-100"}`}>
                      {REASON_LABELS[k as FlagReason] ?? k}: {v}
                    </span>
                  ))}
                </div>
              </div>

              {/* Top unresolved gaps */}
              {report.top_unresolved_gaps.length > 0 && (
                <div>
                  <h2 className="text-base font-semibold text-gray-800 mb-3">
                    Top Unresolved Gaps (by frequency)
                  </h2>
                  <div className="space-y-2">
                    {report.top_unresolved_gaps.map((g) => (
                      <div key={g.id} className="flex items-start gap-3 p-3 border rounded-lg">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">{g.topic}</p>
                          <p className="text-xs text-gray-500 truncate">{g.normalized_question}</p>
                        </div>
                        <span className="text-sm font-bold text-gray-700">×{g.frequency}</span>
                        <span className={`px-2 py-0.5 rounded-full text-xs ${REASON_COLORS[g.flag_reason]}`}>
                          {REASON_LABELS[g.flag_reason]}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <p className="text-xs text-gray-400">
                Report generated: {new Date(report.generated_at).toLocaleString()} · Period: {report.period_days} days
              </p>
            </div>
          ) : null}
        </>
      )}

      {/* Detail / Edit Modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-1">Review Gap #{selected.id}</h3>
            <p className="text-xs text-gray-500 mb-4">
              Topic: <strong>{selected.topic}</strong> · Signal:{" "}
              <span className={`px-1.5 py-0.5 rounded text-xs ${REASON_COLORS[selected.flag_reason]}`}>
                {REASON_LABELS[selected.flag_reason]}
              </span>
            </p>

            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-600 mb-1">Normalized question (de-identified)</label>
              <div className="bg-gray-50 border rounded p-2 text-sm text-gray-700">
                {selected.normalized_question}
              </div>
            </div>

            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-600 mb-1">First seen / last seen / frequency</label>
              <div className="text-sm text-gray-600">
                {new Date(selected.first_seen).toLocaleDateString()} →{" "}
                {new Date(selected.last_seen).toLocaleDateString()} · {selected.frequency}×
              </div>
            </div>

            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-600 mb-1">Status</label>
              <select
                value={editStatus}
                onChange={(e) => setEditStatus(e.target.value)}
                className="w-full border border-gray-300 rounded px-3 py-1.5 text-sm"
              >
                {Object.entries(STATUS_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>
            </div>

            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-600 mb-1">Source reference (optional)</label>
              <input
                type="text"
                value={editRef}
                onChange={(e) => setEditRef(e.target.value)}
                placeholder="e.g. Diccionario Garífuna page 45"
                className="w-full border border-gray-300 rounded px-3 py-1.5 text-sm"
              />
            </div>

            <div className="mb-4">
              <label className="block text-xs font-medium text-gray-600 mb-1">Admin notes</label>
              <textarea
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
                rows={3}
                placeholder="Research notes, next steps…"
                className="w-full border border-gray-300 rounded px-3 py-1.5 text-sm resize-none"
              />
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setSelected(null)}
                className="px-4 py-2 text-sm border rounded text-gray-600 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={saveModal}
                disabled={saving}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: "blue" | "gray";
}) {
  return (
    <div className={`rounded-lg p-4 border ${color === "blue" ? "border-blue-200 bg-blue-50" : "border-gray-200 bg-gray-50"}`}>
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color === "blue" ? "text-blue-700" : "text-gray-800"}`}>
        {value}
      </p>
    </div>
  );
}
