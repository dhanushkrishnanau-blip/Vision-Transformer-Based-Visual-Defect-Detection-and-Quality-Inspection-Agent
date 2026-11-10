// Keep every browser request in one place; the frontend never has database keys.
async function request(path, options = {}) {
  let response
  try {
    response = await fetch(path, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options.headers },
    })
  } catch {
    throw new Error('The backend could not be reached. Make sure it is running.')
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`
    try {
      const body = await response.json()
      if (body.detail) detail = body.detail
    } catch {
      // Some browser or proxy errors do not include a JSON response.
    }
    throw new Error(detail)
  }
  return response
}

async function json(path, options) {
  return (await request(path, options)).json()
}

function queryString(params = {}) {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, value)
  })
  return query.size ? `?${query}` : ''
}

export const getHealth = () => json('/api/health')
export const getOverview = () => json('/api/overview')
export const getInspections = (filters = {}) => json(`/api/inspections${queryString({ limit: 500, ...filters })}`)
export const getInspection = (id) => json(`/api/inspections/${encodeURIComponent(id)}`)
export const getTickets = (filters = {}) => json(`/api/tickets${queryString(filters)}`)
export const getReviews = (pending = false) => json(`/api/reviews${queryString({ pending: pending ? 'true' : '' })}`)
export const getReportInfo = (ticketId) => json(`/api/reports/${encodeURIComponent(ticketId)}`)

export const updateTicket = (id, status) => json(`/api/tickets/${encodeURIComponent(id)}`, {
  method: 'PATCH',
  body: JSON.stringify({ status }),
})

export const resolveReview = (id, reviewer_label, notes) => json(`/api/reviews/${encodeURIComponent(id)}/resolve`, {
  method: 'POST',
  body: JSON.stringify({ reviewer_label, notes }),
})

export const getReport = async (ticketId) => (
  await request(`/api/reports/${encodeURIComponent(ticketId)}/html`)
).blob()
