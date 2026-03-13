// src/services/api.js
const API_BASE_URL = 'https://ecometrics-api.adc.seattle.gov/api';

/**
 * GET /api/health
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/health`);
  const data = await response.json();
  if (!data.status || data.status !== 'healthy') {
    throw new Error('Backend is not healthy');
  }
  return data;
}

/**
 * GET /api/sources
 */
export async function listSources() {
  const response = await fetch(`${API_BASE_URL}/sources`);
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to list sources');
  }
  return data;
}

/**
 * GET /api/sources/{sourceId}/preview
 */
export async function previewSource(sourceId, sheet) {
  let url = `${API_BASE_URL}/sources/${encodeURIComponent(sourceId)}/preview`;
  if (sheet) {
    url += `?sheet=${encodeURIComponent(sheet)}`;
  }
  const response = await fetch(url);
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to preview source');
  }
  return data;
}

/**
 * POST /api/sessions
 */
export async function createSession(name, description) {
  const body = { name };
  if (description) body.description = description;

  const response = await fetch(`${API_BASE_URL}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to create session');
  }
  return data;
}

/**
 * GET /api/sessions
 */
export async function listSessions() {
  const response = await fetch(`${API_BASE_URL}/sessions`);
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to list sessions');
  }
  return data;
}

/**
 * DELETE /api/sessions/{sessionId}
 */
export async function deleteSession(sessionId) {
  const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to delete session');
  }
  return data;
}

/**
 * POST /api/sessions/{sessionId}/sources/{sourceId}/config
 */
export async function saveSourceConfig(sessionId, sourceId, config) {
  const response = await fetch(
    `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/sources/${encodeURIComponent(sourceId)}/config`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }
  );
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to save config');
  }
  return data;
}

/**
 * GET /api/sessions/{sessionId}/sources/{sourceId}/config
 */
export async function getSourceConfig(sessionId, sourceId) {
  const response = await fetch(
    `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/sources/${encodeURIComponent(sourceId)}/config`
  );
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to get config');
  }
  return data;
}

/**
 * POST /api/sessions/{sessionId}/calculate
 */
export async function runCalculation(sessionId, sources, gwpVersion) {
  const body = {};
  if (sources) body.sources = sources;
  if (gwpVersion) body.gwpVersion = gwpVersion;

  const response = await fetch(
    `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/calculate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Calculation failed');
  }
  return data;
}

/**
 * GET /api/sessions/{sessionId}/analytics/summary
 */
export async function getAnalyticsSummary(sessionId) {
  const response = await fetch(
    `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/analytics/summary`
  );
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to get analytics summary');
  }
  return data;
}

/**
 * POST /api/sessions/{sessionId}/analytics/chart-data
 */
export async function getChartData(sessionId, chartType, metric, groupBy, filters) {
  const body = { chartType, metric, groupBy };
  if (filters) body.filters = filters;

  const response = await fetch(
    `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/analytics/chart-data`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to get chart data');
  }
  return data;
}

/**
 * POST /api/files/upload
 * Upload a local file to backend storage (S3 in cloud, local dir in local mode).
 * @param {File} file - The browser File object
 * @param {string} folder - "CONSUMPTION" or "REFERENCE"
 * @returns {{ sourceId, name, fileName, folder, sheets }}
 */
export async function uploadFile(file, folder) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('folder', folder);

  const response = await fetch(`${API_BASE_URL}/files/upload`, {
    method: 'POST',
    body: formData, // No Content-Type header: browser sets multipart boundary automatically
  });
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Upload failed');
  }
  return data;
}

/**
 * Legacy: fetch raw data (kept for backward compatibility)
 */
export async function getRawData(type) {
  const response = await fetch(`${API_BASE_URL}/raw-data?type=${type}`);
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to fetch raw data');
  }
  return data;
}

/**
 * Legacy: fetch calculated data (kept for backward compatibility)
 */
export async function getCalculatedData(sessionId) {
  const response = await fetch(`${API_BASE_URL}/calculated-data?sessionId=${sessionId}`);
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to fetch calculated data');
  }
  return data;
}

export function generateSessionId() {
  return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
