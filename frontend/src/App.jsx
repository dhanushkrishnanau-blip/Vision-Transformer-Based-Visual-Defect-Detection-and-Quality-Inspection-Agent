import { useEffect, useMemo, useState } from 'react'
import {
  getHealth,
  getInspections,
  getInspection,
  getOverview,
  getReport,
  getReportInfo,
  getReviews,
  getTickets,
  resolveReview,
  updateTicket,
} from './api.js'
import './styles.css'

const pages = ['Overview', 'Inspections', 'Tickets', 'Review queue', 'Reports']
const decisions = ['normal', 'uncertain', 'defect']
const statuses = ['open', 'in_progress', 'closed']

function ErrorMessage({ message, onRetry }) {
  if (!message) return null
  return (
    <div className="notice error" role="alert">
      <span>{message}</span>
      {onRetry && <button className="button secondary" onClick={onRetry}>Try again</button>}
    </div>
  )
}

function Loading() {
  return <p className="muted" role="status">Loading dashboard data…</p>
}

function Empty({ children }) {
  return <p className="empty">{children}</p>
}

function StatusBadge({ value }) {
  return <span className={`badge ${value || 'unknown'}`}>{String(value || 'unknown').replace('_', ' ')}</span>
}

function DataTable({ headers, children }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

function Dashboard() {
  const [page, setPage] = useState('Overview')
  const [health, setHealth] = useState(null)
  const [overview, setOverview] = useState(null)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)
  const [inspectionFilter, setInspectionFilter] = useState('all')
  const [selectedInspection, setSelectedInspection] = useState(null)
  const [ticketFilters, setTicketFilters] = useState({ status: 'all', severity: 'all', defect_type: 'all' })
  const [selectedReportId, setSelectedReportId] = useState('')

  const refresh = () => setRefreshKey((key) => key + 1)

  useEffect(() => {
    let active = true
    async function loadHeader() {
      try {
        const [healthData, overviewData] = await Promise.all([getHealth(), getOverview()])
        if (active) {
          setHealth(healthData)
          setOverview(overviewData)
          setError('')
        }
      } catch (requestError) {
        if (active) setError(requestError.message)
      }
    }
    loadHeader()
    return () => { active = false }
  }, [refreshKey])

  const openReport = (ticketId) => {
    setSelectedReportId(String(ticketId))
    setPage('Reports')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setPage('Overview') }}>
          <span className="brand-mark" aria-hidden="true">I</span>
          <span><strong>InspectAI</strong><small>Quality inspection</small></span>
        </a>
        <nav aria-label="Dashboard pages">
          {pages.map((item) => (
            <button
              key={item}
              className={`nav-item ${page === item ? 'active' : ''}`}
              aria-current={page === item ? 'page' : undefined}
              onClick={() => setPage(item)}
            >
              <span className="nav-indicator" aria-hidden="true" />
              {item}
              {item === 'Review queue' && overview?.pending_reviews > 0 && (
                <span className="nav-count">{overview.pending_reviews}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">Project 12 · Industrial inspection</div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><p className="eyebrow">QUALITY OPERATIONS</p><h1>{page}</h1></div>
          <div className="connection-state">
            <span className={`connection-dot ${health ? 'connected' : ''}`} />
            {health ? (health.offline ? 'Offline demo' : 'Backend connected') : 'Connecting'}
          </div>
        </header>

        {health?.offline && (
          <div className="offline-banner" role="status">
            Offline mode is active. Changes are temporary and are not saved to Supabase.
          </div>
        )}
        <ErrorMessage message={error} onRetry={refresh} />

        {page === 'Overview' && (
          <OverviewPage overview={overview} loading={!overview && !error} />
        )}
        {page === 'Inspections' && (
          <InspectionsPage
            filter={inspectionFilter}
            setFilter={setInspectionFilter}
            selectedId={selectedInspection}
            setSelectedId={setSelectedInspection}
            setError={setError}
          />
        )}
        {page === 'Tickets' && (
          <TicketsPage
            filters={ticketFilters}
            setFilters={setTicketFilters}
            setError={setError}
            onReport={openReport}
            refresh={refresh}
          />
        )}
        {page === 'Review queue' && (
          <ReviewPage
            setError={setError}
            refresh={refresh}
          />
        )}
        {page === 'Reports' && (
          <ReportsPage
            selectedId={selectedReportId}
            setSelectedId={setSelectedReportId}
            setError={setError}
          />
        )}
      </main>
    </div>
  )
}

function OverviewPage({ overview, loading }) {
  const cards = overview ? [
    ['Total inspections', overview.total_inspections, 'total'],
    ['Normal', overview.normal, 'normal'],
    ['Uncertain', overview.uncertain, 'uncertain'],
    ['Defect', overview.defect, 'defect'],
    ['Open tickets', overview.open_tickets, 'open'],
    ['Pending reviews', overview.pending_reviews, 'pending'],
    ['Sent to review', `${(overview.share_sent_to_review * 100).toFixed(1)}%`, 'review-share'],
  ] : []
  const maxDecisionCount = overview ? Math.max(overview.normal, overview.uncertain, overview.defect, 1) : 1

  if (loading) return <Loading />
  if (!overview) return <Empty>Overview information is not available.</Empty>

  return (
    <section className="page-section">
      <div className="section-heading">
        <div><p className="eyebrow">LIVE INSPECTION SUMMARY</p><h2>Production at a glance</h2></div>
      </div>
      <div className="metric-grid">
        {cards.map(([label, value, tone]) => (
          <article className={`metric-card tone-${tone}`} key={label}>
            <span>{label}</span><strong>{value}</strong>
          </article>
        ))}
      </div>
      <div className="overview-grid">
        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">CLASSIFICATION</p><h3>Decision split</h3></div></div>
          <div className="bar-list">
            {decisions.map((decision) => {
              const count = overview[decision]
              return (
                <div className="bar-row" key={decision}>
                  <span>{decision}</span>
                  <div className="bar-track" aria-label={`${decision}: ${count}`}>
                    <div className={`bar-fill ${decision}`} style={{ width: `${(count / maxDecisionCount) * 100}%` }} />
                  </div>
                  <strong>{count}</strong>
                </div>
              )
            })}
          </div>
        </article>
        <article className="panel threshold-panel">
          <div className="panel-heading"><div><p className="eyebrow">MODEL SETTINGS</p><h3>Detector thresholds</h3></div></div>
          <div className="threshold-row"><span>Low threshold</span><strong>{Number(overview.thresholds.low_threshold).toFixed(6)}</strong></div>
          <div className="threshold-row"><span>High threshold</span><strong>{Number(overview.thresholds.high_threshold).toFixed(6)}</strong></div>
          <p className="muted small">Scores are compared with these values to support review decisions.</p>
        </article>
      </div>
    </section>
  )
}

function InspectionsPage({ filter, setFilter, selectedId, setSelectedId, setError }) {
  const [inspections, setInspections] = useState([])
  const [ticketsByInspection, setTicketsByInspection] = useState({})
  const [detail, setDetail] = useState(null)
  const [loadingList, setLoadingList] = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)

  useEffect(() => {
    let active = true
    setLoadingList(true)
    Promise.all([
      getInspections(filter === 'all' ? {} : { decision: filter }),
      getTickets({}),
    ])
      .then(([result, tickets]) => {
        if (active) {
          setInspections(result.items)
          setTicketsByInspection(
            Object.fromEntries(tickets.map((ticket) => [String(ticket.inspection_id), ticket])),
          )
        }
      })
      .catch((requestError) => { if (active) setError(requestError.message) })
      .finally(() => { if (active) setLoadingList(false) })
    return () => { active = false }
  }, [filter, setError])

  useEffect(() => {
    if (!selectedId) { setDetail(null); return undefined }
    let active = true
    setLoadingDetail(true)
    getInspection(selectedId)
      .then((result) => { if (active) setDetail(result) })
      .catch((requestError) => { if (active) setError(requestError.message) })
      .finally(() => { if (active) setLoadingDetail(false) })
    return () => { active = false }
  }, [selectedId, setError])

  return (
    <section className="page-section">
      <div className="section-heading">
        <div><p className="eyebrow">IMAGE REVIEW</p><h2>Inspection records</h2></div>
        <label className="filter-control">Decision
          <select value={filter} onChange={(event) => { setSelectedId(null); setFilter(event.target.value) }}>
            <option value="all">All decisions</option>
            {decisions.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
      </div>
      {loadingList ? <Loading /> : inspections.length === 0 ? <Empty>No inspections match this filter.</Empty> : (
        <DataTable headers={['ID', 'Image', 'Score', 'Decision', 'Ticket status']}>
          {inspections.map((inspection) => {
            const ticket = ticketsByInspection[String(inspection.id)]
            return (
              <tr
                key={inspection.id}
                className={`clickable-row ${String(selectedId) === String(inspection.id) ? 'selected-row' : ''}`}
                tabIndex="0"
                onClick={() => setSelectedId(inspection.id)}
                onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') setSelectedId(inspection.id) }}
              >
                <td>{inspection.id}</td>
                <td className="file-name">{inspection.image_path.split('/').pop()}</td>
                <td>{Number(inspection.score).toFixed(4)}</td>
                <td><StatusBadge value={inspection.decision} /></td>
                <td>{ticket ? <StatusBadge value={ticket.status} /> : <span className="muted">—</span>}</td>
              </tr>
            )
          })}
        </DataTable>
      )}
      {selectedId && (
        <div className="detail-panel panel">
          {loadingDetail ? <Loading /> : detail ? (
            <InspectionDetail detail={detail} />
          ) : <Empty>Inspection detail is not available.</Empty>}
        </div>
      )}
    </section>
  )
}

function InspectionDetail({ detail }) {
  const inspection = detail.inspection
  const ticket = detail.ticket || detail.tickets?.[0]
  const review = detail.review || detail.reviews?.[0]
  const score = Number(inspection.score)
  const [thresholdData, setThresholdData] = useState(null)
  useEffect(() => {
    getOverview().then((data) => setThresholdData(data.thresholds)).catch(() => setThresholdData(null))
  }, [])
  const low = thresholdData?.low_threshold ?? 0
  const high = thresholdData?.high_threshold ?? 1
  const max = Math.max(high * 1.25, score, 0.01)

  return (
    <div>
      <div className="section-heading detail-title"><div><p className="eyebrow">INSPECTION #{inspection.id}</p><h2>{inspection.image_path}</h2></div></div>
      <div className="image-pair">
        <figure><figcaption>Original image</figcaption><img src={`/api/images/original/${inspection.id}`} alt="Original inspected product" /></figure>
        <figure><figcaption>Anomaly heatmap</figcaption><img src={`/api/images/heatmap/${inspection.id}`} alt="Heatmap highlighting possible defects" /></figure>
      </div>
      <div className="panel score-panel">
        <div className="panel-heading"><div><p className="eyebrow">MODEL SCORE</p><h3>{score.toFixed(5)}</h3></div><StatusBadge value={inspection.decision} /></div>
        {thresholdData ? (
          <>
            <div className="score-track">
              <span className="score-range" style={{ left: `${(low / max) * 100}%`, width: `${Math.max(0, ((high - low) / max) * 100)}%` }} />
              <span className="score-marker" style={{ left: `${Math.min(100, (score / max) * 100)}%` }} />
            </div>
            <div className="score-legend"><span>0</span><span>Low {low.toFixed(3)}</span><span>High {high.toFixed(3)}</span><span>{max.toFixed(3)}</span></div>
          </>
        ) : <p className="muted">Thresholds could not be loaded.</p>}
      </div>
      <div className="detail-grid">
        <article className="panel"><h3>Ticket</h3>{ticket ? <><p>{ticket.defect_type} · {ticket.severity}</p><StatusBadge value={ticket.status} /><pre>{ticket.procedure_text}</pre></> : <Empty>No ticket linked to this inspection.</Empty>}</article>
        <article className="panel"><h3>Review</h3>{review ? <><p>{review.reason}</p><StatusBadge value={review.reviewer_label || 'pending'} />{review.notes && <p>{review.notes}</p>}</> : <Empty>No review linked to this inspection.</Empty>}</article>
      </div>
      <h3 className="subheading">Agent decision log</h3>
      {(detail.decision_log || []).length === 0 ? <Empty>No log entries are available.</Empty> : (
        <ol className="timeline">
          {detail.decision_log.map((entry) => <li key={entry.id}><time>{entry.created_at}</time><strong>{entry.step} · {entry.tool_name}</strong><p>{entry.input_summary}</p><p>{entry.output_summary}</p></li>)}
        </ol>
      )}
    </div>
  )
}

function TicketsPage({ filters, setFilters, setError, onReport, refresh }) {
  const [tickets, setTickets] = useState([])
  const [loading, setLoading] = useState(true)
  const [draftStatuses, setDraftStatuses] = useState({})
  const [savingId, setSavingId] = useState(null)

  const query = useMemo(() => Object.fromEntries(
    Object.entries(filters).filter(([, value]) => value !== 'all'),
  ), [filters])

  useEffect(() => {
    let active = true
    setLoading(true)
    getTickets(query)
      .then((result) => { if (active) setTickets(result) })
      .catch((requestError) => { if (active) setError(requestError.message) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [query, setError])

  const options = useMemo(() => ({
    severity: [...new Set(tickets.map((ticket) => ticket.severity).filter(Boolean))].sort(),
    defect_type: [...new Set(tickets.map((ticket) => ticket.defect_type).filter(Boolean))].sort(),
  }), [tickets])

  async function saveStatus(ticket) {
    setSavingId(ticket.id)
    try {
      await updateTicket(ticket.id, draftStatuses[ticket.id] || ticket.status)
      setTickets((current) => current.map((row) => row.id === ticket.id ? { ...row, status: draftStatuses[ticket.id] || row.status } : row))
      refresh()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSavingId(null)
    }
  }

  return (
    <section className="page-section">
      <div className="section-heading"><div><p className="eyebrow">CORRECTIVE ACTIONS</p><h2>Ticket management</h2></div></div>
      <div className="filter-row">
        <Filter label="Status" value={filters.status} onChange={(value) => setFilters({ ...filters, status: value })} options={statuses} />
        <Filter label="Severity" value={filters.severity} onChange={(value) => setFilters({ ...filters, severity: value })} options={options.severity} />
        <Filter label="Defect type" value={filters.defect_type} onChange={(value) => setFilters({ ...filters, defect_type: value })} options={options.defect_type} />
      </div>
      {loading ? <Loading /> : tickets.length === 0 ? <Empty>No tickets match these filters.</Empty> : (
        <DataTable headers={['ID', 'Inspection', 'Defect type', 'Severity', 'Status', 'Update status', 'Report']}>
          {tickets.map((ticket) => (
            <tr key={ticket.id}>
              <td>{ticket.id}</td><td>#{ticket.inspection_id}</td><td>{ticket.defect_type}</td><td><StatusBadge value={ticket.severity} /></td><td><StatusBadge value={ticket.status} /></td>
              <td><div className="inline-control">
                <select aria-label={`New status for ticket ${ticket.id}`} value={draftStatuses[ticket.id] || ticket.status} onChange={(event) => setDraftStatuses({ ...draftStatuses, [ticket.id]: event.target.value })}>
                  {statuses.map((status) => <option key={status} value={status}>{status.replace('_', ' ')}</option>)}
                </select>
                <button className="button secondary compact" disabled={savingId === ticket.id} onClick={() => saveStatus(ticket)}>{savingId === ticket.id ? 'Saving…' : 'Save'}</button>
              </div></td>
              <td><button className="button link-button" onClick={() => onReport(ticket.id)}>Open report</button></td>
            </tr>
          ))}
        </DataTable>
      )}
    </section>
  )
}

function Filter({ label, value, onChange, options }) {
  return (
    <label className="filter-control">{label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">All</option>
        {options.map((option) => <option key={option} value={option}>{String(option).replace('_', ' ')}</option>)}
      </select>
    </label>
  )
}

function ReviewPage({ setError, refresh }) {
  const [reviews, setReviews] = useState([])
  const [details, setDetails] = useState({})
  const [notes, setNotes] = useState({})
  const [loading, setLoading] = useState(true)
  const [savingId, setSavingId] = useState(null)

  const loadReviews = async () => {
    setLoading(true)
    try {
      const rows = await getReviews(true)
      setReviews(rows)
      const pairs = await Promise.all(rows.map(async (review) => [review.id, await getInspection(review.inspection_id)]))
      setDetails(Object.fromEntries(pairs))
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadReviews() }, [])

  async function resolve(review, label) {
    setSavingId(review.id)
    try {
      await resolveReview(review.id, label, notes[review.id] || '')
      await loadReviews()
      refresh()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSavingId(null)
    }
  }

  return (
    <section className="page-section">
      <div className="section-heading"><div><p className="eyebrow">HUMAN-IN-THE-LOOP</p><h2>Review queue</h2></div><span className="count-pill">{reviews.length} pending</span></div>
      {loading ? <Loading /> : reviews.length === 0 ? <Empty>There are no pending reviews.</Empty> : (
        <div className="review-list">
          {reviews.map((review) => {
            const inspection = details[review.id]?.inspection
            return (
              <article className="panel review-card" key={review.id}>
                <div className="panel-heading"><div><p className="eyebrow">INSPECTION #{review.inspection_id}</p><h3>{review.reason || 'Manual review requested'}</h3></div><strong>{inspection ? Number(inspection.score).toFixed(5) : '—'}</strong></div>
                {inspection && <div className="image-pair compact-images">
                  <figure><figcaption>Original</figcaption><img src={`/api/images/original/${inspection.id}`} alt="Original image pending review" /></figure>
                  <figure><figcaption>Heatmap</figcaption><img src={`/api/images/heatmap/${inspection.id}`} alt="Heatmap pending review" /></figure>
                </div>}
                <label className="notes-label">Reviewer notes
                  <textarea rows="3" value={notes[review.id] || ''} onChange={(event) => setNotes({ ...notes, [review.id]: event.target.value })} />
                </label>
                <div className="button-row">
                  <button className="button danger" disabled={savingId === review.id} onClick={() => resolve(review, 'defect')}>Confirm defect</button>
                  <button className="button success" disabled={savingId === review.id} onClick={() => resolve(review, 'normal')}>Confirm normal</button>
                </div>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}

function ReportsPage({ selectedId, setSelectedId, setError }) {
  const [tickets, setTickets] = useState([])
  const [loading, setLoading] = useState(true)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportReady, setReportReady] = useState(false)
  const [downloading, setDownloading] = useState(false)
  useEffect(() => {
    let active = true
    getTickets({})
      .then((rows) => { if (active) { setTickets(rows); if (!selectedId && rows.length) setSelectedId(String(rows[0].id)) } })
      .catch((requestError) => { if (active) setError(requestError.message) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [setError, selectedId, setSelectedId])

  const selectedTicket = tickets.find((ticket) => String(ticket.id) === String(selectedId))

  useEffect(() => {
    if (!selectedId) {
      setReportReady(false)
      return undefined
    }
    let active = true
    setReportLoading(true)
    setReportReady(false)
    getReportInfo(selectedId)
      .then(() => { if (active) setReportReady(true) })
      .catch((requestError) => { if (active) setError(requestError.message) })
      .finally(() => { if (active) setReportLoading(false) })
    return () => { active = false }
  }, [selectedId, setError])

  async function download() {
    if (!selectedId) return
    setDownloading(true)
    try {
      const blob = await getReport(selectedId)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `inspection-${selectedTicket?.inspection_id || selectedId}-report.html`
      link.click()
      URL.revokeObjectURL(url)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <section className="page-section">
      <div className="section-heading"><div><p className="eyebrow">AUDIT-READY DOCUMENTS</p><h2>Inspection reports</h2></div></div>
      {loading ? <Loading /> : tickets.length === 0 ? <Empty>No tickets are available for reporting.</Empty> : (
        <>
          <div className="report-toolbar">
            <label className="filter-control">Choose a ticket
              <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
                {tickets.map((ticket) => <option key={ticket.id} value={ticket.id}>Ticket #{ticket.id} · inspection #{ticket.inspection_id} · {ticket.defect_type}</option>)}
              </select>
            </label>
            <button className="button primary" disabled={downloading} onClick={download}>{downloading ? 'Preparing…' : 'Download report'}</button>
          </div>
          {reportLoading && <Loading />}
          {reportReady && <iframe className="report-frame" title="Printable inspection report" src={`/api/reports/${selectedId}/html`} />}
        </>
      )}
    </section>
  )
}

export default Dashboard
