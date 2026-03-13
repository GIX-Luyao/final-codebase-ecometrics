/**
 * API service for EcoMetrics backend integration
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Interface 2: Update data with changes (column renames, cell edits)
 * @param {string} dataType - Type of data ("fleetfuel" or "efid")
 * @param {Array} changes - Array of change objects
 * @param {string|null} sessionId - Optional session ID (auto-generated if not provided)
 * @returns {Promise<Object>} API response
 */
export async function updateData(dataType, changes, sessionId = null) {
  const body = { dataType, changes };
  if (sessionId) {
    body.sessionId = sessionId;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/data/update`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    // Log response for debugging
    console.log('API Response:', data);

    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      success: false,
      error: {
        code: 'NETWORK_ERROR',
        message: error.message || 'Failed to connect to backend'
      }
    };
  }
}

/**
 * Interface 1: Get raw data for preview
 * @param {string} type - Type of data ("fleetfuel" or "efid")
 * @param {boolean} invalid - If true, get invalid test file
 * @returns {Promise<Object>} API response
 */
export async function getRawData(type, invalid = false) {
  const params = new URLSearchParams({ type });
  if (invalid) {
    params.append('invalid', 'true');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/raw-data?${params}`);
    const data = await response.json();
    console.log('Raw Data Response:', data);
    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      success: false,
      error: {
        code: 'NETWORK_ERROR',
        message: error.message || 'Failed to connect to backend'
      }
    };
  }
}

/**
 * Interface 3: Get calculated emissions data
 * @param {string} sessionId - Session ID
 * @returns {Promise<Object>} API response with chart-ready data
 */
export async function getCalculatedData(sessionId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/calculated-data?sessionId=${sessionId}`);
    const data = await response.json();
    console.log('Calculated Data Response:', data);
    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      success: false,
      error: {
        code: 'NETWORK_ERROR',
        message: error.message || 'Failed to connect to backend'
      }
    };
  }
}

/**
 * Health check
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.json();
  } catch (error) {
    return { status: 'error', message: error.message };
  }
}
