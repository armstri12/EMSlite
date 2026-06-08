/**
 * EMSlite API client — fetch() wrappers for all backend endpoints.
 */
const API = {
  async getData(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.panels) q.set("panels", params.panels);
    if (params.department) q.set("department", params.department);
    const res = await fetch("/api/data?" + q.toString());
    return res.json();
  },

  async getMetrics(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.department) q.set("department", params.department);
    const res = await fetch("/api/metrics?" + q.toString());
    return res.json();
  },

  async getDevices(params = {}) {
    const q = new URLSearchParams();
    if (params.department) q.set("department", params.department);
    if (params.enabled_only) q.set("enabled_only", "true");
    const res = await fetch("/api/devices?" + q.toString());
    return res.json();
  },

  async getDevice(id) {
    const res = await fetch("/api/devices/" + encodeURIComponent(id));
    return res.json();
  },

  async updateDevice(id, data) {
    const res = await fetch("/api/devices/" + encodeURIComponent(id), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async bulkAssignDevices(data) {
    const res = await fetch("/api/devices/bulk-assign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async getDepartments() {
    const res = await fetch("/api/departments");
    return res.json();
  },

  async createDepartment(data) {
    const res = await fetch("/api/departments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async updateDepartment(id, data) {
    const res = await fetch("/api/departments/" + encodeURIComponent(id), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteDepartment(id) {
    const res = await fetch("/api/departments/" + encodeURIComponent(id), {
      method: "DELETE",
    });
    return res.json();
  },

  // ── Floor Plans ──
  async getFloorPlans() {
    const res = await fetch("/api/floorplans");
    return res.json();
  },

  async getFloorPlan(id) {
    const res = await fetch("/api/floorplans/" + id);
    return res.json();
  },

  async createFloorPlan(formData) {
    const res = await fetch("/api/floorplans", { method: "POST", body: formData });
    return res.json();
  },

  async updateFloorPlan(id, data) {
    const res = await fetch("/api/floorplans/" + id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteFloorPlan(id) {
    const res = await fetch("/api/floorplans/" + id, { method: "DELETE" });
    return res.json();
  },

  async addFloorPlanPin(planId, data) {
    const res = await fetch("/api/floorplans/" + planId + "/pins", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async updateFloorPlanPin(planId, pinId, data) {
    const res = await fetch("/api/floorplans/" + planId + "/pins/" + pinId, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteFloorPlanPin(planId, pinId) {
    const res = await fetch("/api/floorplans/" + planId + "/pins/" + pinId, { method: "DELETE" });
    return res.json();
  },

  async addFloorPlanZone(planId, data) {
    const res = await fetch("/api/floorplans/" + planId + "/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async updateFloorPlanZone(planId, zoneId, data) {
    const res = await fetch("/api/floorplans/" + planId + "/zones/" + zoneId, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteFloorPlanZone(planId, zoneId) {
    const res = await fetch("/api/floorplans/" + planId + "/zones/" + zoneId, { method: "DELETE" });
    return res.json();
  },

  // ── Utility Bills ──
  async getBills(meter) {
    const q = new URLSearchParams();
    if (meter) q.set("meter", meter);
    const res = await fetch("/api/bills?" + q.toString());
    return res.json();
  },

  async createBill(data) {
    const res = await fetch("/api/bills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async updateBill(id, data) {
    const res = await fetch("/api/bills/" + id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteBill(id) {
    const res = await fetch("/api/bills/" + id, { method: "DELETE" });
    return res.json();
  },

  async getBillComparison(id) {
    const res = await fetch("/api/bills/" + id + "/comparison");
    return res.json();
  },

  async getMeterCoverage({ start, end } = {}) {
    const q = new URLSearchParams();
    if (start) q.set("start", start);
    if (end) q.set("end", end);
    const res = await fetch("/api/meter-coverage?" + q.toString());
    return res.json();
  },

  async getConfig() {
    const res = await fetch("/api/config");
    return res.json();
  },

  async updateConfig(data) {
    const res = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async getIngestLog(limit = 50) {
    const res = await fetch("/api/ingest-log?limit=" + limit);
    return res.json();
  },

  async getAlerts(params = {}) {
    const q = new URLSearchParams();
    if (params.includeAcknowledged) q.set("include_acknowledged", "true");
    if (params.windowHours) q.set("window_hours", String(params.windowHours));
    if (params.latestOnly) q.set("latest_only", "true");
    const res = await fetch("/api/alerts?" + q.toString());
    if (!res.ok) throw new Error(`Failed to fetch alerts: ${res.status}`);
    return res.json();
  },

  async acknowledgeAlerts(keys) {
    const res = await fetch("/api/alerts/ack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keys }),
    });
    if (!res.ok) throw new Error(`Failed to acknowledge alerts: ${res.status}`);
    return res.json();
  },

  async getDataHealth() {
    const res = await fetch("/api/data-health");
    return res.json();
  },

  async getWeather(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    const res = await fetch("/api/weather?" + q.toString());
    return res.json();
  },

  // ─── HVAC / weather correlation ───
  async getHvacTemperatures(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.unit) q.set("unit", params.unit);
    const res = await fetch("/api/hvac/temperatures?" + q.toString());
    if (!res.ok) throw new Error(`getHvacTemperatures failed: ${res.status}`);
    return res.json();
  },

  async uploadHvacTemperatures(data) {
    const res = await fetch("/api/hvac/temperatures/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      let detail = res.status;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(`Upload failed: ${detail}`);
    }
    return res.json();
  },

  async clearHvacTemperatures() {
    const res = await fetch("/api/hvac/temperatures", { method: "DELETE" });
    if (!res.ok) throw new Error(`clearHvacTemperatures failed: ${res.status}`);
    return res.json();
  },

  async getHvacAnalysis(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.unit) q.set("unit", params.unit);
    if (params.balance_point !== undefined && params.balance_point !== "")
      q.set("balance_point", String(params.balance_point));
    if (params.panels && params.panels.length) q.set("panels", params.panels.join(","));
    const res = await fetch("/api/hvac/analysis?" + q.toString());
    if (!res.ok) throw new Error(`getHvacAnalysis failed: ${res.status}`);
    return res.json();
  },

  async getBehaviorRankings(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    const res = await fetch("/api/behavior/rankings?" + q.toString());
    if (!res.ok) throw new Error(`Rankings failed: ${res.status}`);
    return res.json();
  },

  async getBehavior(params = {}) {
    const q = new URLSearchParams();
    if (params.panel) q.set("panel", params.panel);
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    const res = await fetch("/api/behavior?" + q.toString());
    if (!res.ok) throw new Error(`Behavior analysis failed: ${res.status}`);
    return res.json();
  },

  async getTrendingSnapshot(params = {}) {
    const q = new URLSearchParams();
    if (params.periodDays) q.set("period_days", String(params.periodDays));
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    const res = await fetch("/api/trending/snapshot?" + q.toString());
    if (!res.ok) throw new Error(`Trending snapshot failed: ${res.status}`);
    return res.json();
  },

  async getTrendingDetail(params = {}) {
    const q = new URLSearchParams();
    if (params.panel) q.set("panel", params.panel);
    if (params.periodDays) q.set("period_days", String(params.periodDays));
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    const res = await fetch("/api/trending/detail?" + q.toString());
    if (!res.ok) throw new Error(`Trending detail failed: ${res.status}`);
    return res.json();
  },

  // ─── Production metrics ───
  async getMetricDefinitions() {
    const res = await fetch("/api/metric-definitions");
    return res.json();
  },

  async createMetricDefinition(data) {
    const res = await fetch("/api/metric-definitions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`createMetricDefinition failed: ${res.status}`);
    return res.json();
  },

  async updateMetricDefinition(id, data) {
    const res = await fetch("/api/metric-definitions/" + encodeURIComponent(id), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteMetricDefinition(id) {
    const res = await fetch("/api/metric-definitions/" + encodeURIComponent(id), {
      method: "DELETE",
    });
    return res.json();
  },

  async getDailyMetrics(params = {}) {
    const q = new URLSearchParams();
    if (params.metric_def_id) q.set("metric_def_id", params.metric_def_id);
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    const res = await fetch("/api/daily-metrics?" + q.toString());
    return res.json();
  },

  async createDailyMetric(data) {
    const res = await fetch("/api/daily-metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`createDailyMetric failed: ${res.status}`);
    return res.json();
  },

  async updateDailyMetric(id, data) {
    const res = await fetch("/api/daily-metrics/" + id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteDailyMetric(id) {
    const res = await fetch("/api/daily-metrics/" + id, { method: "DELETE" });
    return res.json();
  },

  async bulkUploadDailyMetrics(data) {
    const res = await fetch("/api/daily-metrics/bulk-upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  // ─── Workflows ───
  async getWorkflows() {
    const res = await fetch("/api/workflows");
    return res.json();
  },

  async getWorkflow(id) {
    const res = await fetch("/api/workflows/" + id);
    return res.json();
  },

  async createWorkflow(data) {
    const res = await fetch("/api/workflows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`createWorkflow failed: ${res.status}`);
    return res.json();
  },

  async updateWorkflow(id, data) {
    const res = await fetch("/api/workflows/" + id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async deleteWorkflow(id) {
    const res = await fetch("/api/workflows/" + id, { method: "DELETE" });
    return res.json();
  },

  async getProductionCorrelation(params = {}) {
    const q = new URLSearchParams();
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.metric_def_ids && params.metric_def_ids.length)
      q.set("metric_def_ids", params.metric_def_ids.join(","));
    if (params.panels && params.panels.length)
      q.set("panels", params.panels.join(","));
    const res = await fetch("/api/production/correlation?" + q.toString());
    return res.json();
  },

  // ── Wireless Sensors ──
  async getWirelessStatus() {
    const res = await fetch("/api/wireless/status");
    return res.json();
  },

  async getWirelessGateways() {
    const res = await fetch("/api/wireless/gateways");
    return res.json();
  },

  async getWirelessSensors(params = {}) {
    const q = new URLSearchParams();
    if (params.gateway_id) q.set("gateway_id", params.gateway_id);
    if (params.enabled_only) q.set("enabled_only", "true");
    const res = await fetch("/api/wireless/sensors?" + q.toString());
    return res.json();
  },

  async getWirelessSensor(id) {
    const res = await fetch("/api/wireless/sensors/" + encodeURIComponent(id));
    return res.json();
  },

  async updateWirelessSensor(id, data) {
    const res = await fetch("/api/wireless/sensors/" + encodeURIComponent(id), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async getWirelessReadings(params = {}) {
    const q = new URLSearchParams();
    if (params.sensor_id) q.set("sensor_id", params.sensor_id);
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.limit) q.set("limit", String(params.limit));
    const res = await fetch("/api/wireless/readings?" + q.toString());
    return res.json();
  },

  // ─── Demand Response ───
  async getDRPrograms() {
    const res = await fetch("/api/demand-response/programs");
    return res.json();
  },

  async createDRProgram(data) {
    const res = await fetch("/api/demand-response/programs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`createDRProgram failed: ${res.status}`);
    return res.json();
  },

  async updateDRProgram(id, data) {
    const res = await fetch("/api/demand-response/programs/" + id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`updateDRProgram failed: ${res.status}`);
    return res.json();
  },

  async deleteDRProgram(id) {
    const res = await fetch("/api/demand-response/programs/" + id, { method: "DELETE" });
    return res.json();
  },

  async getDREvents(params = {}) {
    const q = new URLSearchParams();
    if (params.program_id) q.set("program_id", String(params.program_id));
    if (params.status) q.set("status", params.status);
    const res = await fetch("/api/demand-response/events?" + q.toString());
    return res.json();
  },

  async createDREvent(data) {
    const res = await fetch("/api/demand-response/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`createDREvent failed: ${res.status}`);
    return res.json();
  },

  async updateDREvent(id, data) {
    const res = await fetch("/api/demand-response/events/" + id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`updateDREvent failed: ${res.status}`);
    return res.json();
  },

  async deleteDREvent(id) {
    const res = await fetch("/api/demand-response/events/" + id, { method: "DELETE" });
    return res.json();
  },

  async computeDREvent(id) {
    const res = await fetch("/api/demand-response/events/" + id + "/compute", { method: "POST" });
    if (!res.ok) throw new Error(`computeDREvent failed: ${res.status}`);
    return res.json();
  },

  async getDREventProfile(id) {
    const res = await fetch("/api/demand-response/events/" + id + "/profile");
    return res.json();
  },

  async getDRSeasonSummary(programId) {
    const res = await fetch("/api/demand-response/season-summary/" + programId);
    return res.json();
  },

  getEmailSummaryUrl(panels, start, end, priorStart, priorEnd, download = false) {
    const q = new URLSearchParams();
    if (panels && panels.length) q.set("panels", panels.join(","));
    if (start) q.set("start", start);
    if (end) q.set("end", end);
    if (priorStart) q.set("prior_start", priorStart);
    if (priorEnd) q.set("prior_end", priorEnd);
    if (download) q.set("download", "true");
    return "/api/reports/email-summary?" + q.toString();
  },
};
