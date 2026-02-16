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
};
