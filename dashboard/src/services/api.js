/**
 * SecureIoT-X — API Service Layer.
 * Handles all HTTP + WebSocket communication with the FastAPI backend.
 *
 * Docker: VITE_API_BASE=/api (nginx proxies to backend container)
 * Local:  falls back to http://localhost:8000/api
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api';

// Auto-detect WebSocket URL: if API is relative (Docker), use same host
const _isRelative = API_BASE.startsWith('/');
const WS_BASE = _isRelative
  ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
  : 'ws://localhost:8000/ws';

class APIService {
  constructor() {
    this.token = localStorage.getItem('secureiot_token');
    this.ws = null;
    this.wsListeners = new Map();
  }

  // ── Auth Headers ────────────────────────────────────────
  get headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  }

  // ── Generic Fetch ───────────────────────────────────────
  async request(path, options = {}) {
    try {
      const resp = await fetch(`${API_BASE}${path}`, {
        headers: this.headers,
        ...options,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      return resp.json();
    } catch (e) {
      console.error(`API error [${path}]:`, e.message);
      throw e;
    }
  }

  // ── Auth ────────────────────────────────────────────────
  async login(username, password) {
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    this.token = data.access_token;
    localStorage.setItem('secureiot_token', data.access_token);
    return data;
  }

  logout() {
    this.token = null;
    localStorage.removeItem('secureiot_token');
  }

  // ── Health ──────────────────────────────────────────────
  async getHealth() { return this.request('/health'); }

  // ── Devices ─────────────────────────────────────────────
  async getDevices() { return this.request('/devices'); }
  async registerDevice(deviceData) {
    return this.request('/devices', {
      method: 'POST',
      body: JSON.stringify(deviceData),
    });
  }
  async isolateDevice(id) { return this.request(`/devices/${id}/isolate`, { method: 'POST' }); }
  async releaseDevice(id) { return this.request(`/devices/${id}/release`, { method: 'POST' }); }
  async deleteDevice(id) { return this.request(`/devices/${id}`, { method: 'DELETE' }); }
  async getEnforcementDetails(id) { return this.request(`/devices/${id}/enforcement`); }
  async probeTestEndpoint(id) { return this.request(`/devices/probe/test?device_id=${id}`); }
  async runDetectionTest(id) { return this.request(`/devices/${id}/test/detection`, { method: 'POST' }); }
  async runEnforcementTest(id) { return this.request(`/devices/${id}/test/enforcement`, { method: 'POST' }); }
  async getDeviceTelemetry(deviceId, limit = 50) {
    return this.request(`/devices/${deviceId}/telemetry?limit=${limit}`);
  }
  async getRecentTelemetry(limit = 50) {
    return this.request(`/devices/telemetry/recent?limit=${limit}`);
  }

  // ── Predictions ─────────────────────────────────────────
  async predict(deviceId, features) {
    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify({ device_id: deviceId, features }),
    });
  }
  async getPredictions(limit = 50) { return this.request(`/predictions?limit=${limit}`); }

  // ── Threats ─────────────────────────────────────────────
  async getThreats(limit = 50) { return this.request(`/threats?limit=${limit}`); }

  // ── Blockchain ──────────────────────────────────────────
  async getBlockchainStatus() { return this.request('/blockchain/status'); }
  async getBlocks(limit = 20) { return this.request(`/blockchain/blocks?limit=${limit}`); }
  async validateChain() { return this.request('/blockchain/verify', { method: 'POST' }); }
  async tamperTest() { return this.request('/blockchain/tamper-test', { method: 'POST' }); }

  // ── Analytics ───────────────────────────────────────────
  async getOverview() { return this.request('/analytics/overview'); }
  async getAIMetrics() { return this.request('/analytics/metrics'); }
  async getModelComparison() { return this.request('/analytics/comparison'); }

  // ── Models ──────────────────────────────────────────────
  async getModels() { return this.request('/models'); }
  async getModelReport() { return this.request('/models/report'); }
  async activateModel(name, version = 'v1') {
    return this.request(`/models/${name}/activate?version=${version}`, { method: 'POST' });
  }

  // ── Simulation ──────────────────────────────────────────
  async simulateAttack(deviceId, attackType) {
    return this.request('/simulate/attack', {
      method: 'POST',
      body: JSON.stringify({ device_id: deviceId, attack_type: attackType }),
    });
  }
  async releaseSimDevice(deviceId) {
    return this.request('/simulate/release', {
      method: 'POST',
      body: JSON.stringify({ device_id: deviceId }),
    });
  }
  async getSimDevices() { return this.request('/simulate/devices'); }
  async simulateEvent(type = 'normal') {
    return this.request(`/simulate/event?event_type=${type}`, { method: 'POST' });
  }
  async runScenario(scenario) {
    return this.request('/simulate/scenario', {
      method: 'POST',
      body: JSON.stringify({ scenario }),
    });
  }
  async getScenarios() { return this.request('/simulate/scenarios'); }

  // ── Ganache ─────────────────────────────────────────────
  async getGanacheStatus() { return this.request('/blockchain/ganache/status'); }
  async verifyGanacheTx(txHash) { return this.request(`/blockchain/ganache/verify/${txHash}`); }

  // ── System ──────────────────────────────────────────────
  async getSystemLogs(limit = 100) { return this.request(`/system/logs?limit=${limit}`); }
  async getAuditTrail(limit = 100) { return this.request(`/system/audit?limit=${limit}`); }
  async getSystemSettings() { return this.request('/system/settings'); }

  // ── WebSocket ───────────────────────────────────────────
  connectWS() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(WS_BASE);

    this.ws.onopen = () => console.log('[WS] Connected');

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const listeners = this.wsListeners.get(data.type) || [];
        listeners.forEach(cb => cb(data.data));
      } catch (e) {
        console.warn('[WS] Parse error:', e);
      }
    };

    this.ws.onclose = () => {
      console.log('[WS] Disconnected, reconnecting in 3s...');
      setTimeout(() => this.connectWS(), 3000);
    };

    this.ws.onerror = (err) => console.error('[WS] Error:', err);
  }

  onWS(type, callback) {
    if (!this.wsListeners.has(type)) this.wsListeners.set(type, []);
    this.wsListeners.get(type).push(callback);
  }

  offWS(type) {
    this.wsListeners.delete(type);
  }
}

export const api = new APIService();
export default api;
