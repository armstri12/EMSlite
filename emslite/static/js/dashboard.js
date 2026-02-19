/**
 * EMSlite Dashboard — main rendering logic.
 *
 * Adapted from the inline JS in energy_dashboard.py.
 * Data is fetched from the API instead of being embedded.
 */

/* ─── Global State ─── */
let D = { timestamps: [], total_kw: [], panel_series: {}, panel_names: [],
           group_series: {}, group_definitions: [], utility_meters: [],
           meter_series: {}, rolling_hours: 1, price_per_kwh: 0.25 };
let PRICE = 0.25;
let ALL_PANELS = [];
let DATE_MIN = "";
let DATE_MAX = "";
let isDark = false;
let activeTab = "executive";
let DEPARTMENTS = [];       // [{id, display_name, color, device_count}]
let DEPT_DEVICE_MAP = {};   // department_id -> [panel_id, ...]
let ALL_DEVICES = [];       // [{id, display_name, ...}]
let ACTIVE_ALERTS = [];
let ALERT_SUMMARY = { critical: 0, warning: 0, configured_devices: 0, window_hours: 24 };
let ALERTS_INCLUDE_ACK = false;
let ALERTS_WINDOW_HOURS = 24;
let WEATHER = { enabled: false, timestamps: [], temperature_c: [], humidity_pct: [], unit: "celsius" };
let WEATHER_OVERLAY = { temperature: false, humidity: false };

/* ─── Theme Palettes — Edwards brand ─── */
const lightC = {
  ink:"#3A424D", muted:"#9BA5B0", card:"#ffffff", bg:"#F8FAFB",
  grid:"rgba(0,0,0,0.06)", accent:"#C41230", accentDark:"#2D3842",
  gradFrom:"#2D3842", gradTo:"#C41230", donutA:"#C41230", donutB:"#2D3842",
  series:["#C41230","#2D3842","#398EB9","#F97316","#16A34A","#38BDF8","#9333EA","#1D5B83"]
};
const darkC = {
  ink:"#E3E8EC", muted:"#6C7784", card:"#171C23", bg:"#0D1117",
  grid:"rgba(255,255,255,0.06)", accent:"#C41230", accentDark:"#E3E8EC",
  gradFrom:"#2D3842", gradTo:"#C41230", donutA:"#C41230", donutB:"#4E6373",
  series:["#C41230","#38BDF8","#F97316","#16A34A","#74B6D5","#9333EA","#D4EAF2","#F5939D"]
};
function T() { return isDark ? darkC : lightC; }

/* ─── Init ─── */
async function initDashboard() {
  try {
    const [data, departments, devices] = await Promise.all([
      API.getData(),
      API.getDepartments(),
      API.getDevices(),
    ]);
    D = data;
    PRICE = data.price_per_kwh || 0.25;
    ALL_PANELS = data.panel_names || [];
    DATE_MIN = data.timestamps.length ? new Date(data.timestamps[0]).toISOString().slice(0,10) : "";
    DATE_MAX = data.timestamps.length ? new Date(data.timestamps[data.timestamps.length-1]).toISOString().slice(0,10) : "";

    // Build department & device state
    DEPARTMENTS = departments || [];
    ALL_DEVICES = devices || [];
    DEPT_DEVICE_MAP = {};
    for (const dept of DEPARTMENTS) {
      DEPT_DEVICE_MAP[dept.id] = [];
    }
    for (const dev of ALL_DEVICES) {
      if (dev.department_id && DEPT_DEVICE_MAP[dev.department_id]) {
        DEPT_DEVICE_MAP[dev.department_id].push(dev.id);
      }
    }

    // Load weather config and data if enabled
    try {
      const cfg = await API.getConfig();
      const wCfg = (cfg && cfg.weather) || {};
      if (wCfg.enabled && wCfg.station_id && DATE_MIN && DATE_MAX) {
        WEATHER = await API.getWeather({ start: DATE_MIN, end: DATE_MAX });
      }
    } catch (e) {
      console.warn("Weather data load skipped:", e);
    }

    initTabState();
    setupThemeToggle();
    setupSidebar();
    setupTabNavigation();
    setupNotifications();
    // Overview has no filter bar — it auto-displays latest data
    buildFilterBar("an-filter-bar", "analytics",  "daterange");
    buildFilterBar("cp-filter-bar", "comparison", "comparison");
    buildFilterBar("dt-filter-bar", "data",       "daterange");
    buildBehaviorFilterBar();
    setupTableControls();
    renderCurrentTab();
    await refreshAlerts(true);
  } catch (err) {
    console.error("Failed to load dashboard data:", err);
    document.querySelector(".main-container").innerHTML =
      '<div style="padding:2rem;text-align:center;color:var(--muted)">' +
      '<h2>No data available</h2><p>Drop CSV files into the <code>drops/</code> folder to get started.</p></div>';
  }
}

/* ═══════════════════════════════════════════════════════
   PER-TAB STATE
   ═══════════════════════════════════════════════════════ */
const tabState = {
  overview:   { panels: new Set(), startDate: "", endDate: "", department: "" },
  analytics:  { panels: new Set(), startDate: "", endDate: "", department: "" },
  comparison: { panels: new Set(), p1Start: "", p1End: "", p2Start: "", p2End: "", department: "" },
  data:       { panels: new Set(), startDate: "", endDate: "", department: "" },
  behavior:   { panel: "", startDate: "", endDate: "" }
};

function initTabState() {
  ["overview","analytics","comparison","data"].forEach(tab => {
    tabState[tab].panels = new Set(ALL_PANELS);
  });
  tabState.overview.startDate = DATE_MIN;
  tabState.overview.endDate = DATE_MAX;
  tabState.analytics.startDate = DATE_MIN;
  tabState.analytics.endDate = DATE_MAX;
  tabState.data.startDate = DATE_MIN;
  tabState.data.endDate = DATE_MAX;
  tabState.behavior.startDate = DATE_MIN;
  tabState.behavior.endDate = DATE_MAX;
  tabState.behavior.panel = ALL_PANELS.length ? ALL_PANELS[0] : "";

  // Auto-init comparison periods
  const totalDays = DATE_MIN && DATE_MAX ? (new Date(DATE_MAX) - new Date(DATE_MIN)) / 86400000 : 0;
  const st = tabState.comparison;
  if (totalDays >= 14) {
    const e2 = new Date(DATE_MAX), s2 = new Date(e2); s2.setDate(s2.getDate()-6);
    const e1 = new Date(s2); e1.setDate(e1.getDate()-1);
    const s1 = new Date(e1); s1.setDate(s1.getDate()-6);
    st.p1Start = s1.toISOString().slice(0,10);
    st.p1End   = e1.toISOString().slice(0,10);
    st.p2Start = s2.toISOString().slice(0,10);
    st.p2End   = e2.toISOString().slice(0,10);
  } else if (totalDays > 0) {
    const mid = new Date(DATE_MIN); mid.setDate(mid.getDate()+Math.floor(totalDays/2));
    const midN = new Date(mid); midN.setDate(midN.getDate()+1);
    st.p1Start = DATE_MIN;
    st.p1End   = mid.toISOString().slice(0,10);
    st.p2Start = midN.toISOString().slice(0,10);
    st.p2End   = DATE_MAX;
  }
}

/* ═══════════════════════════════════════════════════════
   SETUP FUNCTIONS
   ═══════════════════════════════════════════════════════ */

function setupThemeToggle() {
  document.getElementById("theme-toggle").addEventListener("click", () => {
    isDark = !isDark;
    document.documentElement.classList.toggle("dark", isDark);
    document.getElementById("theme-label").textContent = isDark ? "Light" : "Dark";
    document.getElementById("theme-icon").innerHTML = isDark
      ? '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
      : '<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>';
    renderCurrentTab();
  });
}

function setupSidebar() {
  const sidebar = document.getElementById("sidebar");
  const sidebarOverlay = document.getElementById("sidebar-overlay");
  document.getElementById("hamburger").addEventListener("click", () => {
    sidebar.classList.toggle("open");
    sidebarOverlay.classList.toggle("active");
  });
  sidebarOverlay.addEventListener("click", () => {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("active");
  });
}

function setupTabNavigation() {
  document.querySelectorAll(".nav-item[data-tab]").forEach(n => {
    n.addEventListener("click", e => {
      e.preventDefault();
      switchTab(n.dataset.tab);
      document.getElementById("sidebar").classList.remove("open");
      document.getElementById("sidebar-overlay").classList.remove("active");
    });
  });
}

function setupNotifications() {
  const btn = document.getElementById("notifications-btn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    switchTab("alerts");
  });
}

async function refreshAlerts(latestOnly = false) {
  try {
    const payload = await API.getAlerts({
      includeAcknowledged: ALERTS_INCLUDE_ACK,
      windowHours: ALERTS_WINDOW_HOURS,
      latestOnly,
    });
    ACTIVE_ALERTS = payload.alerts || [];
    ALERT_SUMMARY = payload.summary || ALERT_SUMMARY;
  } catch (_e) {
    ACTIVE_ALERTS = [];
  }
  const dot = document.querySelector(".notif-dot");
  if (dot) dot.style.display = ACTIVE_ALERTS.some(a => !a.acknowledged) ? "block" : "none";
}

function showNotificationsPanel() {
  const overlay = document.createElement("div");
  overlay.className = "sidebar-overlay active";
  overlay.style.zIndex = "400";
  const panel = document.createElement("div");
  panel.className = "fp-editor";
  panel.style.zIndex = "401";
  panel.innerHTML = `
    <div class="fp-editor-head">
      <h3>Active Alerts (${ACTIVE_ALERTS.length})</h3>
      <button class="btn btn-ghost btn-sm" id="close-alerts">Close</button>
    </div>
    <div style="display:flex;flex-direction:column;gap:0.5rem;margin-top:1rem;max-height:70vh;overflow:auto;">
      ${ACTIVE_ALERTS.length ? ACTIVE_ALERTS.map(a => `
        <div style="border:1px solid var(--card-border);border-radius:10px;padding:0.75rem;background:var(--card-bg)">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem">
            <div>
              <div style="font-weight:600;color:var(--heading)">${a.device_name}</div>
              <div style="font-size:0.75rem;color:var(--muted)">${a.timestamp}</div>
            </div>
            <span class="badge ${a.severity === 'critical' ? 'badge-neg' : 'badge-warn'}">${a.severity}</span>
          </div>
          <div style="margin-top:0.5rem;font-size:0.875rem">${a.current_kw.toFixed(2)} kW (threshold: ${a.threshold_kw} kW)</div>
          <button class="btn btn-ghost btn-sm ack-alert" data-key="${a.key}" style="margin-top:0.5rem">Acknowledge</button>
        </div>
      `).join("") : `<div style="color:var(--muted)">No active alerts.</div>`}
    </div>`;
  document.body.appendChild(overlay);
  document.body.appendChild(panel);
  const close = () => { overlay.remove(); panel.remove(); };
  overlay.addEventListener("click", close);
  panel.querySelector("#close-alerts").addEventListener("click", close);
  panel.querySelectorAll(".ack-alert").forEach(btn => {
    btn.addEventListener("click", async () => {
      await API.acknowledgeAlerts([btn.dataset.key]);
      await refreshAlerts(true);
      close();
      showNotificationsPanel();
    });
  });
}

function switchTab(tabId) {
  activeTab = tabId;
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".nav-item[data-tab]").forEach(n => n.classList.remove("active"));
  const tabEl = document.getElementById("tab-" + tabId);
  if (tabEl) tabEl.classList.add("active");
  document.querySelectorAll('.nav-item[data-tab="' + tabId + '"]').forEach(n => n.classList.add("active"));
  renderCurrentTab();
  window.dispatchEvent(new Event("resize"));
}

function setupTableControls() {
  document.getElementById("export-csv").addEventListener("click", () => {
    const st = tabState.data;
    const panels = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
    let csv = ["Timestamp","Total_kW",...panels].join(",") + "\n";
    tData.forEach(r => { csv += r.join(",") + "\n"; });
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = "energy_export.csv"; a.click(); URL.revokeObjectURL(a.href);
  });
  document.getElementById("table-search").addEventListener("input", () => { tPage = 0; renderTableRows(); });
}

/* ═══════════════════════════════════════════════════════
   FILTER BAR BUILDER
   ═══════════════════════════════════════════════════════ */
function buildFilterBar(containerId, tabKey, mode) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const st = tabState[tabKey];
  const uid = tabKey;

  let html = '';

  // Department filter
  if (DEPARTMENTS.length) {
    html += `<div class="filter-group"><label>Department</label>
      <select id="dept-${uid}" class="filter-input">
        <option value="">All Departments</option>
        ${DEPARTMENTS.map(d => `<option value="${d.id}"${st.department === d.id ? ' selected' : ''}>${d.display_name}</option>`).join("")}
      </select></div>
    <div class="filter-divider"></div>`;
  }

  if (ALL_PANELS.length) {
    html += `<div class="dropdown-wrap" id="dd-${uid}">
      <button class="dropdown-trigger" id="ddt-${uid}">Panels <span class="dd-badge" id="ddb-${uid}">All</span></button>
      <div class="dropdown-menu" id="ddm-${uid}">
        <div class="dd-actions">
          <button id="dda-${uid}">Select All</button>
          <button id="ddn-${uid}">Clear</button>
        </div>
        <div class="dd-list" id="ddl-${uid}"></div>
      </div>
    </div>
    <div class="filter-divider"></div>`;
  }

  if (mode === "daterange") {
    html += `<div class="filter-group"><label>From</label>
      <input type="date" id="fs-${uid}" class="filter-input" value="${st.startDate}" min="${DATE_MIN}" max="${DATE_MAX}"/></div>
    <div class="filter-group"><label>To</label>
      <input type="date" id="fe-${uid}" class="filter-input" value="${st.endDate}" min="${DATE_MIN}" max="${DATE_MAX}"/></div>`;
  } else if (mode === "comparison") {
    html += `<div class="filter-group"><label>Period 1 Start</label>
      <input type="date" id="cp1s-${uid}" class="filter-input" value="${st.p1Start}"/></div>
    <div class="filter-group"><label>Period 1 End</label>
      <input type="date" id="cp1e-${uid}" class="filter-input" value="${st.p1End}"/></div>
    <div class="filter-divider"></div>
    <div class="filter-group"><label>Period 2 Start</label>
      <input type="date" id="cp2s-${uid}" class="filter-input" value="${st.p2Start}"/></div>
    <div class="filter-group"><label>Period 2 End</label>
      <input type="date" id="cp2e-${uid}" class="filter-input" value="${st.p2End}"/></div>`;
  }

  // Weather overlay toggles (Analytics & Comparison tabs only)
  if (WEATHER.enabled && (tabKey === "analytics" || tabKey === "comparison")) {
    html += `<div class="filter-divider"></div>
    <div class="filter-group" style="gap:10px;align-items:center">
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer;white-space:nowrap">
        <input type="checkbox" id="wx-temp-${uid}" ${WEATHER_OVERLAY.temperature ? "checked" : ""}
               style="accent-color:#F97316"/>
        <span style="color:#F97316;font-size:0.7rem;font-weight:600">&#9728;</span>
        <span style="font-size:0.75rem">Temp</span>
      </label>
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer;white-space:nowrap">
        <input type="checkbox" id="wx-hum-${uid}" ${WEATHER_OVERLAY.humidity ? "checked" : ""}
               style="accent-color:#38BDF8"/>
        <span style="color:#38BDF8;font-size:0.7rem;font-weight:600">&#9679;</span>
        <span style="font-size:0.75rem">Humidity</span>
      </label>
    </div>`;
  }

  html += `<button class="btn btn-primary" id="apply-${uid}">Apply</button>
    <button class="btn btn-ghost" id="reset-${uid}">Reset</button>`;

  container.innerHTML = html;

  // Wire panel dropdown
  if (ALL_PANELS.length) {
    const list = document.getElementById("ddl-" + uid);
    ALL_PANELS.forEach(p => {
      const lbl = document.createElement("label"); lbl.className = "dd-item";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.value = p; cb.checked = st.panels.has(p);
      cb.addEventListener("change", () => { cb.checked ? st.panels.add(p) : st.panels.delete(p); syncDDBadge(uid, st); });
      const sp = document.createElement("span"); sp.textContent = p;
      lbl.appendChild(cb); lbl.appendChild(sp); list.appendChild(lbl);
    });
    document.getElementById("ddt-" + uid).addEventListener("click", e => {
      e.stopPropagation(); document.getElementById("ddm-" + uid).classList.toggle("open");
    });
    document.addEventListener("click", e => {
      const wrap = document.getElementById("dd-" + uid);
      if (wrap && !wrap.contains(e.target)) document.getElementById("ddm-" + uid).classList.remove("open");
    });
    document.getElementById("dda-" + uid).addEventListener("click", () => {
      st.panels = new Set(ALL_PANELS);
      document.querySelectorAll("#ddl-" + uid + " input").forEach(c => c.checked = true);
      syncDDBadge(uid, st);
    });
    document.getElementById("ddn-" + uid).addEventListener("click", () => {
      st.panels.clear();
      document.querySelectorAll("#ddl-" + uid + " input").forEach(c => c.checked = false);
      syncDDBadge(uid, st);
    });
    syncDDBadge(uid, st);
  }

  // Wire department dropdown
  const deptSel = document.getElementById("dept-" + uid);
  if (deptSel) {
    deptSel.addEventListener("change", () => {
      st.department = deptSel.value;
      // Auto-filter panels to match department
      if (st.department && DEPT_DEVICE_MAP[st.department]) {
        const deptPanels = new Set(DEPT_DEVICE_MAP[st.department]);
        st.panels = new Set(ALL_PANELS.filter(p => deptPanels.has(p)));
      } else {
        st.panels = new Set(ALL_PANELS);
      }
      // Sync panel checkboxes
      document.querySelectorAll("#ddl-" + uid + " input").forEach(c => {
        c.checked = st.panels.has(c.value);
      });
      syncDDBadge(uid, st);
      renderTab(tabKey);
    });
  }

  // Wire Apply
  document.getElementById("apply-" + uid).addEventListener("click", () => {
    if (mode === "daterange") {
      st.startDate = document.getElementById("fs-" + uid).value || DATE_MIN;
      st.endDate   = document.getElementById("fe-" + uid).value || DATE_MAX;
    } else {
      st.p1Start = document.getElementById("cp1s-" + uid).value;
      st.p1End   = document.getElementById("cp1e-" + uid).value;
      st.p2Start = document.getElementById("cp2s-" + uid).value;
      st.p2End   = document.getElementById("cp2e-" + uid).value;
    }
    renderTab(tabKey);
  });

  // Wire Reset
  document.getElementById("reset-" + uid).addEventListener("click", () => {
    st.department = "";
    const deptReset = document.getElementById("dept-" + uid);
    if (deptReset) deptReset.value = "";
    st.panels = new Set(ALL_PANELS);
    document.querySelectorAll("#ddl-" + uid + " input").forEach(c => c.checked = true);
    syncDDBadge(uid, st);
    if (mode === "daterange") {
      st.startDate = DATE_MIN; st.endDate = DATE_MAX;
      document.getElementById("fs-" + uid).value = DATE_MIN;
      document.getElementById("fe-" + uid).value = DATE_MAX;
    } else {
      const totalDays = DATE_MIN && DATE_MAX ? (new Date(DATE_MAX) - new Date(DATE_MIN)) / 86400000 : 0;
      if (totalDays >= 14) {
        const e2 = new Date(DATE_MAX), s2 = new Date(e2); s2.setDate(s2.getDate()-6);
        const e1 = new Date(s2); e1.setDate(e1.getDate()-1);
        const s1 = new Date(e1); s1.setDate(s1.getDate()-6);
        st.p1Start = s1.toISOString().slice(0,10); st.p1End = e1.toISOString().slice(0,10);
        st.p2Start = s2.toISOString().slice(0,10); st.p2End = e2.toISOString().slice(0,10);
      } else {
        const mid = new Date(DATE_MIN); mid.setDate(mid.getDate()+Math.floor(totalDays/2));
        const midN = new Date(mid); midN.setDate(midN.getDate()+1);
        st.p1Start = DATE_MIN; st.p1End = mid.toISOString().slice(0,10);
        st.p2Start = midN.toISOString().slice(0,10); st.p2End = DATE_MAX;
      }
      document.getElementById("cp1s-" + uid).value = st.p1Start;
      document.getElementById("cp1e-" + uid).value = st.p1End;
      document.getElementById("cp2s-" + uid).value = st.p2Start;
      document.getElementById("cp2e-" + uid).value = st.p2End;
    }
    renderTab(tabKey);
  });

  // Wire weather overlay toggles
  const wxTemp = document.getElementById("wx-temp-" + uid);
  const wxHum = document.getElementById("wx-hum-" + uid);
  if (wxTemp) {
    wxTemp.addEventListener("change", () => {
      WEATHER_OVERLAY.temperature = wxTemp.checked;
      renderTab(tabKey);
    });
  }
  if (wxHum) {
    wxHum.addEventListener("change", () => {
      WEATHER_OVERLAY.humidity = wxHum.checked;
      renderTab(tabKey);
    });
  }
}

function syncDDBadge(uid, st) {
  const el = document.getElementById("ddb-" + uid);
  if (el) el.textContent = st.panels.size === ALL_PANELS.length ? "All" : st.panels.size === 0 ? "None" : st.panels.size;
}

/* ═══════════════════════════════════════════════════════
   PER-TAB DATA FILTERING
   ═══════════════════════════════════════════════════════ */
function filterForTab(tabKey) {
  const st = tabState[tabKey];
  const sD = st.startDate ? new Date(st.startDate + "T00:00:00Z") : null;
  const eD = st.endDate   ? new Date(st.endDate   + "T23:59:59Z") : null;
  const active = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  const aSet = new Set(active);
  const gMap = {}, mMap = {};
  (D.group_definitions||[]).forEach(g => { gMap[g.name] = (g.panels||[]).filter(p=>aSet.has(p)); });
  (D.utility_meters||[]).forEach(m => { mMap[m.name] = (m.panels||[]).filter(p=>aSet.has(p)); });
  const f = { timestamps:[], totalKw:[],
    groupSeries: Object.fromEntries(Object.keys(D.group_series||{}).map(k=>[k,[]])),
    meterSeries: Object.fromEntries(Object.keys(D.meter_series||{}).map(k=>[k,[]])),
    panelSeries: Object.fromEntries(ALL_PANELS.map(k=>[k,[]]))
  };
  (D.timestamps||[]).forEach((ts,i) => {
    const d = new Date(ts);
    if (sD && d < sD) return; if (eD && d > eD) return;
    f.timestamps.push(ts);
    let tot=0; active.forEach(p => { tot += (D.panel_series[p]||[])[i]||0; }); f.totalKw.push(tot);
    Object.keys(D.group_series||{}).forEach(g => {
      let s=0; (gMap[g]||[]).forEach(p => { s += (D.panel_series[p]||[])[i]||0; }); f.groupSeries[g].push(s);
    });
    Object.keys(D.meter_series||{}).forEach(m => {
      let s=0; (mMap[m]||[]).forEach(p => { s += (D.panel_series[p]||[])[i]||0; }); f.meterSeries[m].push(s);
    });
    ALL_PANELS.forEach(p => { f.panelSeries[p].push((D.panel_series[p]||[])[i]||0); });
  });
  return f;
}

/* ═══════════════════════════════════════════════════════
   SHARED HELPERS
   ═══════════════════════════════════════════════════════ */
function metrics(ts, kw) {
  let kwh=0, pk=0, sum=0;
  for (let i=0;i<ts.length;i++) { const v=kw[i]??0; sum+=v; if(v>pk)pk=v;
    if(i>0) { const h=Math.max(0,(new Date(ts[i])-new Date(ts[i-1]))/3600000); kwh+=v*h; }
  }
  return { totalKwh:kwh, avgKw:ts.length?sum/ts.length:0, peakKw:pk };
}
function rollingMean(ts,v,wH) {
  const wMs=wH*3600000, r=[]; let si=0,s=0;
  for(let i=0;i<ts.length;i++) { const ct=new Date(ts[i]).getTime(); s+=v[i]??0;
    while(ct-new Date(ts[si]).getTime()>wMs){ s-=v[si]??0;si++; }
    r.push((i-si+1)?s/(i-si+1):0);
  } return r;
}
function dailyEnergy(ts,kw) {
  const ed={};
  for(let i=0;i<ts.length-1;i++) { const h=Math.max(0,(new Date(ts[i+1])-new Date(ts[i]))/3600000);
    const dk=new Date(ts[i]).toISOString().slice(0,10); if(!ed[dk])ed[dk]=0; ed[dk]+=(kw[i]??0)*h;
  } const dates=Object.keys(ed).sort(); return { dates, values:dates.map(d=>ed[d]) };
}
function hourlyProfile(ts,kw) {
  const s=Array(24).fill(0),c=Array(24).fill(0);
  ts.forEach((t,i)=>{ const h=new Date(t).getUTCHours(); s[h]+=kw[i]??0; c[h]++; });
  return { hours:Array.from({length:24},(_,i)=>i), avgs:s.map((v,i)=>c[i]?v/c[i]:0) };
}
function weekdayProfile(ts,kw) {
  const s=Array(7).fill(0),c=Array(7).fill(0);
  ts.forEach((t,i)=>{ const w=new Date(t).getUTCDay(); s[w]+=kw[i]??0; c[w]++; });
  return { days:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], avgs:s.map((v,i)=>c[i]?v/c[i]:0) };
}
function sparkSVG(vals, color) {
  if(!vals.length) return "";
  const w=200,h=32, mn=Math.min(...vals), mx=Math.max(...vals), rng=mx-mn||1;
  const step=w/Math.max(vals.length-1,1);
  const pts=vals.map((v,i)=>`${(i*step).toFixed(1)},${(h-((v-mn)/rng)*h*0.8-h*0.1).toFixed(1)}`);
  const poly=pts.join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polygon points="0,${h} ${poly} ${((vals.length-1)*step).toFixed(1)},${h}" fill="${color}" opacity="0.12"/>
    <polyline points="${poly}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}

/* ─── Plotly helpers ─── */
function pLayout(ov) {
  const t=T();
  return Object.assign({ margin:{t:16,l:55,r:20,b:45}, paper_bgcolor:t.card, plot_bgcolor:t.card,
    font:{family:"Inter,system-ui,sans-serif",color:t.ink,size:12}, colorway:t.series,
    hovermode:"x unified", hoverlabel:{bgcolor:t.ink,font:{color:t.card}} }, ov);
}
function xA(ov) { const t=T(); return Object.assign({gridcolor:t.grid,zerolinecolor:t.grid,showline:true,linecolor:t.grid},ov); }
function yA(ov) { const t=T(); return Object.assign({rangemode:"tozero",gridcolor:t.grid,zerolinecolor:t.grid,showline:true,linecolor:t.grid},ov); }
const pCfg = {displaylogo:false, responsive:true};
function weekendShapes(ts) {
  if(!ts.length)return[];const shapes=[];let ws=null;
  for(let i=0;i<ts.length;i++){ const d=new Date(ts[i]).getUTCDay();const isWe=d===0||d===6;
    if(isWe&&ws===null)ws=ts[i]; if(!isWe&&ws!==null){ shapes.push({type:"rect",xref:"x",yref:"paper",x0:ws,x1:ts[i],y0:0,y1:1,line:{width:0},fillcolor:"rgba(139,212,53,0.06)"}); ws=null; }
  } if(ws!==null)shapes.push({type:"rect",xref:"x",yref:"paper",x0:ws,x1:ts[ts.length-1],y0:0,y1:1,line:{width:0},fillcolor:"rgba(139,212,53,0.06)"}); return shapes;
}

/* ─── Weather overlay helpers ─── */

function filterWeather(startISO, endISO) {
  if (!WEATHER.enabled || !WEATHER.timestamps.length) return null;
  const sD = startISO ? new Date(startISO) : null;
  const eD = endISO ? new Date(endISO) : null;
  const ts = [], temp = [], hum = [];
  WEATHER.timestamps.forEach((t, i) => {
    const d = new Date(t);
    if (sD && d < sD) return;
    if (eD && d > eD) return;
    ts.push(t);
    temp.push(WEATHER.temperature_c[i]);
    hum.push(WEATHER.humidity_pct[i]);
  });
  if (!ts.length) return null;
  return { timestamps: ts, temperature_c: temp, humidity_pct: hum };
}

function weatherTimeSeriesTraces(wx, opts) {
  opts = opts || {};
  const traces = [];
  const dash = opts.dash || "dot";
  const unitLabel = WEATHER.unit === "fahrenheit" ? "\u00B0F" : "\u00B0C";
  if (WEATHER_OVERLAY.temperature && wx.temperature_c.some(v => v != null)) {
    traces.push({
      x: wx.timestamps, y: wx.temperature_c,
      mode: "lines", name: "Temp (" + unitLabel + ")",
      line: { color: "#F97316", width: 1.5, dash: dash, shape: "spline" },
      yaxis: "y2", opacity: 0.85,
      hovertemplate: "%{y:.1f}" + unitLabel + "<extra>Temp</extra>"
    });
  }
  if (WEATHER_OVERLAY.humidity && wx.humidity_pct.some(v => v != null)) {
    traces.push({
      x: wx.timestamps, y: wx.humidity_pct,
      mode: "lines", name: "Humidity (%)",
      line: { color: "#38BDF8", width: 1.5, dash: dash, shape: "spline" },
      yaxis: "y3", opacity: 0.75,
      hovertemplate: "%{y:.0f}%<extra>Humidity</extra>"
    });
  }
  return traces;
}

function weatherAxesLayout() {
  const layout = {};
  if (WEATHER_OVERLAY.temperature) {
    layout.yaxis2 = {
      title: { text: WEATHER.unit === "fahrenheit" ? "\u00B0F" : "\u00B0C",
               font: { size: 11, color: "#F97316" } },
      overlaying: "y", side: "right", showgrid: false, zeroline: false,
      tickfont: { color: "#F97316", size: 10 }
    };
  }
  if (WEATHER_OVERLAY.humidity) {
    layout.yaxis3 = {
      title: { text: "%RH", font: { size: 11, color: "#38BDF8" } },
      overlaying: "y", side: "right",
      position: WEATHER_OVERLAY.temperature ? 0.97 : 1.0,
      showgrid: false, zeroline: false,
      tickfont: { color: "#38BDF8", size: 10 }
    };
    if (WEATHER_OVERLAY.temperature) layout.margin = { r: 75 };
  }
  return layout;
}

function weatherDailyTraces(wx) {
  const traces = [];
  const dayT = {}, dayH = {}, dayC = {};
  wx.timestamps.forEach((t, i) => {
    const dk = new Date(t).toISOString().slice(0, 10);
    if (!dayT[dk]) { dayT[dk] = 0; dayH[dk] = 0; dayC[dk] = 0; }
    if (wx.temperature_c[i] != null) { dayT[dk] += wx.temperature_c[i]; dayC[dk]++; }
    if (wx.humidity_pct[i] != null) dayH[dk] += wx.humidity_pct[i];
  });
  const dates = Object.keys(dayT).sort();
  const unitLabel = WEATHER.unit === "fahrenheit" ? "\u00B0F" : "\u00B0C";
  if (WEATHER_OVERLAY.temperature) {
    traces.push({
      x: dates, y: dates.map(d => dayC[d] ? dayT[d] / dayC[d] : null),
      mode: "lines+markers", name: "Avg Temp (" + unitLabel + ")",
      line: { color: "#F97316", width: 2, shape: "spline" },
      marker: { size: 5, color: "#F97316" }, yaxis: "y2"
    });
  }
  if (WEATHER_OVERLAY.humidity) {
    traces.push({
      x: dates, y: dates.map(d => dayC[d] ? dayH[d] / dayC[d] : null),
      mode: "lines+markers", name: "Avg Humidity (%)",
      line: { color: "#38BDF8", width: 2, shape: "spline" },
      marker: { size: 5, color: "#38BDF8" }, yaxis: "y3"
    });
  }
  return traces;
}

function weatherHourlyProfileTraces(wx) {
  const traces = [];
  const sT = Array(24).fill(0), sH = Array(24).fill(0), c = Array(24).fill(0);
  wx.timestamps.forEach((t, i) => {
    const h = new Date(t).getUTCHours();
    if (wx.temperature_c[i] != null) { sT[h] += wx.temperature_c[i]; c[h]++; }
    if (wx.humidity_pct[i] != null) sH[h] += wx.humidity_pct[i];
  });
  const hours = Array.from({ length: 24 }, (_, i) => i);
  const unitLabel = WEATHER.unit === "fahrenheit" ? "\u00B0F" : "\u00B0C";
  if (WEATHER_OVERLAY.temperature) {
    traces.push({
      x: hours, y: hours.map(h => c[h] ? sT[h] / c[h] : null),
      mode: "lines+markers", name: "Avg Temp",
      line: { color: "#F97316", width: 1.5, dash: "dot", shape: "spline" },
      marker: { size: 5, color: "#F97316" }, yaxis: "y2"
    });
  }
  if (WEATHER_OVERLAY.humidity) {
    traces.push({
      x: hours, y: hours.map(h => c[h] ? sH[h] / c[h] : null),
      mode: "lines+markers", name: "Avg Humidity",
      line: { color: "#38BDF8", width: 1.5, dash: "dot", shape: "spline" },
      marker: { size: 5, color: "#38BDF8" }, yaxis: "y3"
    });
  }
  return traces;
}

function hasWeatherOverlay() {
  return WEATHER.enabled && (WEATHER_OVERLAY.temperature || WEATHER_OVERLAY.humidity);
}

/* ═══════════════════════════════════════════════════════
   TAB RENDERERS
   ═══════════════════════════════════════════════════════ */

async function renderOverview() {
  const t = T();
  const ts = D.timestamps || [];
  const totalKw = D.total_kw || [];
  const panelSeries = D.panel_series || {};
  const groupSeries = D.group_series || {};
  const groupNames = D.group_names || [];
  const now = ts.length ? new Date(ts[ts.length - 1]) : new Date();

  const fmtKwh = v => v >= 10000 ? (v/1000).toFixed(1) + " MWh" : v >= 100 ? v.toFixed(0) + " kWh" : v.toFixed(1) + " kWh";
  const fmtDollars = v => v >= 10000 ? "$" + (v/1000).toFixed(1) + "k" : v >= 1 ? "$" + v.toFixed(0) : "$" + v.toFixed(2);
  const fmtDollarsBig = v => v >= 10000 ? "$" + (v/1000).toFixed(1) + "k" : "$" + v.toFixed(2);
  const fmtPct = (a, b) => { if (!b) return "\u2014"; const d = ((a - b) / b * 100); return (d >= 0 ? "+" : "") + d.toFixed(0) + "%"; };
  const trendClass = (a, b) => a > b ? "trend-up" : a < b ? "trend-down" : "trend-flat";
  const fmtDate = d => d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  const fmtRange = (a, b) => fmtDate(a) + " \u2013 " + fmtDate(b);
  const fmtMonth = d => d.toLocaleDateString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
  const fmtKw = v => v >= 1000 ? (v/1000).toFixed(1) + " MW" : v.toFixed(0) + " kW";

  // ── Compute kWh over a date window ──
  function windowKwh(series, from, to) {
    let kwh = 0;
    for (let i = 1; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      if (ti < from || ti > to) continue;
      const h = Math.max(0, (ti - new Date(ts[i-1])) / 3600000);
      kwh += (series[i] ?? 0) * h;
    }
    return kwh;
  }
  function windowAvgKw(series, from, to) {
    let sum = 0, cnt = 0;
    for (let i = 0; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      if (ti < from || ti > to) continue;
      sum += series[i] ?? 0; cnt++;
    }
    return cnt ? sum / cnt : 0;
  }
  function windowPeakKw(series, from, to) {
    let pk = 0;
    for (let i = 0; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      if (ti < from || ti > to) continue;
      const v = series[i] ?? 0;
      if (v > pk) pk = v;
    }
    return pk;
  }

  // ── Time period boundaries ──
  // "Last Full Week" = most recently completed Mon–Sun
  const dayOfWeek = now.getUTCDay(); // 0=Sun
  const mondayOffset = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const thisWeekStart = new Date(now); thisWeekStart.setUTCDate(thisWeekStart.getUTCDate() - mondayOffset); thisWeekStart.setUTCHours(0,0,0,0);
  const lastWeekEnd = new Date(thisWeekStart); lastWeekEnd.setUTCMilliseconds(-1); // Sun 23:59:59
  const lastWeekStart = new Date(lastWeekEnd); lastWeekStart.setUTCDate(lastWeekStart.getUTCDate() - 6); lastWeekStart.setUTCHours(0,0,0,0);
  // "Week before that" = the Mon–Sun before last full week
  const prevWeekEnd = new Date(lastWeekStart); prevWeekEnd.setUTCMilliseconds(-1);
  const prevWeekStart = new Date(prevWeekEnd); prevWeekStart.setUTCDate(prevWeekStart.getUTCDate() - 6); prevWeekStart.setUTCHours(0,0,0,0);

  // "Last Weekend" — most recent completed Sat + Sun
  const lastSat = new Date(now);
  while (lastSat.getUTCDay() !== 6) lastSat.setUTCDate(lastSat.getUTCDate() - 1);
  // If today is Saturday, go back one more week to get the *completed* weekend
  if (now.getUTCDay() === 6 || now.getUTCDay() === 0) {
    // We might be on the current weekend, take the prior one
    if (lastSat >= thisWeekStart) lastSat.setUTCDate(lastSat.getUTCDate() - 7);
  }
  lastSat.setUTCHours(0,0,0,0);
  const lastSunEnd = new Date(lastSat); lastSunEnd.setUTCDate(lastSunEnd.getUTCDate() + 1); lastSunEnd.setUTCHours(23,59,59,999);

  // "Weekend Before That"
  const prevSat = new Date(lastSat); prevSat.setUTCDate(prevSat.getUTCDate() - 7);
  const prevSunEnd = new Date(prevSat); prevSunEnd.setUTCDate(prevSunEnd.getUTCDate() + 1); prevSunEnd.setUTCHours(23,59,59,999);

  // "Month-to-date"
  const mtdStart = new Date(now); mtdStart.setUTCDate(1); mtdStart.setUTCHours(0,0,0,0);
  // "Previous month"
  const prevMonthEnd = new Date(mtdStart); prevMonthEnd.setUTCMilliseconds(-1);
  const prevMonthStart = new Date(prevMonthEnd); prevMonthStart.setUTCDate(1); prevMonthStart.setUTCHours(0,0,0,0);

  // ── Facility-level period metrics ──
  const lastFullWeekKwh = windowKwh(totalKw, lastWeekStart, lastWeekEnd);
  const prevFullWeekKwh = windowKwh(totalKw, prevWeekStart, prevWeekEnd);
  const lastWeekendKwh = windowKwh(totalKw, lastSat, lastSunEnd);
  const prevWeekendKwh = windowKwh(totalKw, prevSat, prevSunEnd);
  const mtdKwh = windowKwh(totalKw, mtdStart, now);
  const prevMonthKwh = windowKwh(totalKw, prevMonthStart, prevMonthEnd);
  const lastWeekPeak = windowPeakKw(totalKw, lastWeekStart, lastWeekEnd);
  const prevWeekPeakVal = windowPeakKw(totalKw, prevWeekStart, prevWeekEnd);
  const lastWeekAvg = windowAvgKw(totalKw, lastWeekStart, lastWeekEnd);
  const prevWeekAvgVal = windowAvgKw(totalKw, prevWeekStart, prevWeekEnd);

  // Costs
  const lastFullWeekCost = lastFullWeekKwh * PRICE;
  const prevFullWeekCost = prevFullWeekKwh * PRICE;
  const lastWeekendCost = lastWeekendKwh * PRICE;
  const prevWeekendCost = prevWeekendKwh * PRICE;
  const mtdCost = mtdKwh * PRICE;

  // ── Department breakdown (Last Full Week) ──
  const deptColors = ["#C41230","#2D3842","#398EB9","#F97316","#16A34A","#38BDF8","#9333EA","#1D5B83"];
  const disabledDevices = new Set(ALL_DEVICES.filter(d => !d.enabled).map(d => d.id));

  // Build dept->color lookup for top consumers
  const deptColorMap = {};
  let deptRows = [];
  if (groupNames.length) {
    deptRows = groupNames.map((name, idx) => {
      const series = groupSeries[name] || [];
      const kwh = windowKwh(series, lastWeekStart, lastWeekEnd);
      const avgKw = windowAvgKw(series, lastWeekStart, lastWeekEnd);
      const peakKw = windowPeakKw(series, lastWeekStart, lastWeekEnd);
      const prevKwh = windowKwh(series, prevWeekStart, prevWeekEnd);
      const label = name.replace(/_kW$/i, "").replace(/_/g, " ");
      const color = deptColors[idx % deptColors.length];
      deptColorMap[name] = color;
      return { name: label, rawName: name, color, kwh, prevKwh, avgKw, peakKw };
    }).filter(d => d.kwh > 0 || d.prevKwh > 0);
  } else if (DEPARTMENTS.length) {
    deptRows = DEPARTMENTS.map((dept, idx) => {
      const panels = (DEPT_DEVICE_MAP[dept.id] || []).filter(p => !disabledDevices.has(p));
      let kwh = 0, prevKwh = 0, avgKw = 0, peakKw = 0;
      panels.forEach(p => {
        const s = panelSeries[p] || [];
        kwh += windowKwh(s, lastWeekStart, lastWeekEnd);
        prevKwh += windowKwh(s, prevWeekStart, prevWeekEnd);
        avgKw += windowAvgKw(s, lastWeekStart, lastWeekEnd);
        peakKw = Math.max(peakKw, windowPeakKw(s, lastWeekStart, lastWeekEnd));
      });
      const color = dept.color || deptColors[idx % deptColors.length];
      deptColorMap[dept.id] = color;
      return { name: dept.display_name, rawName: dept.id, color, kwh, prevKwh, avgKw, peakKw };
    }).filter(d => d.kwh > 0 || d.prevKwh > 0);
    const assignedPanels = new Set();
    Object.values(DEPT_DEVICE_MAP).forEach(arr => arr.forEach(p => assignedPanels.add(p)));
    const unassigned = ALL_PANELS.filter(p => !assignedPanels.has(p) && !disabledDevices.has(p));
    if (unassigned.length) {
      let kwh = 0, prevKwh = 0, avgKw = 0, peakKw = 0;
      unassigned.forEach(p => {
        const s = panelSeries[p] || [];
        kwh += windowKwh(s, lastWeekStart, lastWeekEnd);
        prevKwh += windowKwh(s, prevWeekStart, prevWeekEnd);
        avgKw += windowAvgKw(s, lastWeekStart, lastWeekEnd);
        peakKw = Math.max(peakKw, windowPeakKw(s, lastWeekStart, lastWeekEnd));
      });
      if (kwh > 0 || prevKwh > 0) {
        deptRows.push({ name: "Unassigned", rawName: "unassigned", color: "#9BA5B0", kwh, prevKwh, avgKw, peakKw });
      }
    }
  }
  deptRows.sort((a, b) => b.kwh - a.kwh);
  const deptTotal = deptRows.reduce((s, d) => s + d.kwh, 0) || 1;
  const deptMax = deptRows.length ? deptRows[0].kwh : 1;

  // ── Top consumers by kWh last full week — skip disabled ──
  // Find department color for each device
  function getDeviceDeptColor(devId) {
    const dev = ALL_DEVICES.find(d => d.id === devId);
    if (!dev) return "#9BA5B0";
    if (dev.department_id && deptColorMap[dev.department_id]) return deptColorMap[dev.department_id];
    // Check group membership
    for (const [gName, gColor] of Object.entries(deptColorMap)) {
      const gDef = (D.group_definitions || []).find(g => g.name === gName);
      if (gDef && gDef.panels && gDef.panels.includes(devId)) return gColor;
    }
    return "#9BA5B0";
  }

  const topConsumers = ALL_PANELS
    .filter(p => !disabledDevices.has(p))
    .map(p => {
      const kwh = windowKwh(panelSeries[p] || [], lastWeekStart, lastWeekEnd);
      const dev = ALL_DEVICES.find(d => d.id === p);
      const color = getDeviceDeptColor(p);
      return { name: dev ? dev.display_name : p, kwh, cost: kwh * PRICE, color };
    }).sort((a, b) => b.kwh - a.kwh).slice(0, 5);
  const topMax = topConsumers.length ? topConsumers[0].kwh : 1;

  // ── Build TOP cards (above floor plan) ──
  const topCardsHtml = `
    <div class="exec-cards-grid cols-4">
      <div class="exec-card exec-headline">
        <div class="exec-section-label">Last Full Week <span class="exec-date-range">${fmtRange(lastWeekStart, lastWeekEnd)}</span></div>
        <div class="exec-headline-cost">${fmtDollarsBig(lastFullWeekCost)}</div>
        <div class="exec-headline-val">${fmtKwh(lastFullWeekKwh)}</div>
        <div class="exec-headline-sub">
          <span class="${trendClass(lastFullWeekKwh, prevFullWeekKwh)}">${fmtPct(lastFullWeekKwh, prevFullWeekKwh)}</span> vs prior week (${fmtDollars(prevFullWeekCost)})
        </div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Last Weekend</div>
        <div class="exec-kpi-cost">${fmtDollars(lastWeekendCost)}</div>
        <div class="exec-kpi-val">${fmtKwh(lastWeekendKwh)}</div>
        <div class="exec-kpi-sub"><span class="${trendClass(lastWeekendKwh, prevWeekendKwh)}">${fmtPct(lastWeekendKwh, prevWeekendKwh)}</span> vs prior</div>
        <div class="exec-kpi-dates">${fmtRange(lastSat, lastSunEnd)}</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Weekend Before</div>
        <div class="exec-kpi-cost">${fmtDollars(prevWeekendCost)}</div>
        <div class="exec-kpi-val">${fmtKwh(prevWeekendKwh)}</div>
        <div class="exec-kpi-dates">${fmtRange(prevSat, prevSunEnd)}</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Month to Date</div>
        <div class="exec-kpi-cost">${fmtDollars(mtdCost)}</div>
        <div class="exec-kpi-val">${fmtKwh(mtdKwh)}</div>
        <div class="exec-kpi-sub"><span class="${trendClass(mtdKwh, prevMonthKwh)}">${fmtPct(mtdKwh, prevMonthKwh)}</span> vs ${fmtMonth(prevMonthStart)}</div>
        <div class="exec-kpi-dates">${fmtRange(mtdStart, now)}</div>
      </div>
    </div>`;

  document.getElementById("exec-cards-top").innerHTML = topCardsHtml;

  // ── Build BOTTOM cards (below floor plan) ──
  const bottomCardsHtml = `
    <div class="exec-cards-grid cols-3">
      <div class="exec-card">
        <div class="exec-section-label">Load Summary <span class="exec-date-range">${fmtRange(lastWeekStart, lastWeekEnd)}</span></div>
        <div class="exec-dept-row">
          <div class="exec-dept-name">Peak Load</div>
          <div class="exec-dept-val" style="width:auto">${fmtKw(lastWeekPeak)}</div>
          <div class="exec-dept-stats"><span class="${trendClass(lastWeekPeak, prevWeekPeakVal)}">${fmtPct(lastWeekPeak, prevWeekPeakVal)}</span> vs prior wk</div>
        </div>
        <div class="exec-dept-row">
          <div class="exec-dept-name">Avg Load</div>
          <div class="exec-dept-val" style="width:auto">${fmtKw(lastWeekAvg)}</div>
          <div class="exec-dept-stats"><span class="${trendClass(lastWeekAvg, prevWeekAvgVal)}">${fmtPct(lastWeekAvg, prevWeekAvgVal)}</span> vs prior wk</div>
        </div>
        <div class="exec-dept-row" style="margin-top:4px;padding-top:4px;border-top:1px solid var(--card-border)">
          <div class="exec-dept-name" style="color:var(--muted);font-size:0.72rem">Load Factor</div>
          <div class="exec-dept-val" style="width:auto">${lastWeekPeak > 0 ? (lastWeekAvg / lastWeekPeak * 100).toFixed(0) + "%" : "\u2014"}</div>
        </div>
      </div>
      ${deptRows.length ? `<div class="exec-card">
        <div class="exec-section-label">Departments <span class="exec-date-range">Last Full Week</span></div>
        ${deptRows.map(d => `<div class="exec-dept-row">
          <div class="exec-dept-dot" style="background:${d.color}"></div>
          <div class="exec-dept-name">${d.name}</div>
          <div class="exec-dept-bar-wrap"><div class="exec-dept-bar-fill" style="width:${(d.kwh/deptMax*100).toFixed(0)}%;background:${d.color}"></div></div>
          <div class="exec-dept-cost">${fmtDollars(d.kwh * PRICE)}</div>
          <div class="exec-dept-val">${fmtKwh(d.kwh)}</div>
        </div>`).join("")}
        <div style="margin-top:6px;padding-top:6px;border-top:1px solid var(--card-border);display:flex;flex-wrap:wrap;gap:10px">
          ${deptRows.map(d => `<div style="font-size:0.62rem;color:var(--muted)">
            <span class="exec-dept-dot" style="background:${d.color};display:inline-block;width:6px;height:6px;vertical-align:middle"></span>
            Avg ${fmtKw(d.avgKw)} / Peak ${fmtKw(d.peakKw)}
          </div>`).join("")}
        </div>
      </div>` : '<div class="exec-card"><div class="exec-section-label">Departments</div><div style="color:var(--muted);font-size:0.78rem">No department data</div></div>'}
      ${topConsumers.length ? `<div class="exec-card">
        <div class="exec-section-label">Top Consumers <span class="exec-date-range">Last Full Week</span></div>
        ${topConsumers.map(c => `<div class="exec-top-row">
          <div class="exec-top-dot" style="background:${c.color}"></div>
          <div class="exec-top-name">${c.name}</div>
          <div class="exec-top-bar-wrap"><div class="exec-top-bar-fill" style="width:${(c.kwh/(topMax||1)*100).toFixed(0)}%;background:${c.color}"></div></div>
          <div class="exec-top-cost">${fmtDollars(c.cost)}</div>
          <div class="exec-top-val">${fmtKwh(c.kwh)}</div>
        </div>`).join("")}
      </div>` : ""}
    </div>`;

  document.getElementById("exec-cards-bottom").innerHTML = bottomCardsHtml;

  // ── Floor plan (full width, center section) ──
  await _renderExecFloorPlan(panelSeries, groupSeries, groupNames, ts, now, fmtKwh, fmtDollars, fmtPct, trendClass, t, windowKwh, windowAvgKw, windowPeakKw, lastWeekStart, lastWeekEnd, lastSat, lastSunEnd, prevSat, prevSunEnd);
}

async function renderExecutive() {
  const ts = D.timestamps || [];
  const totalKw = D.total_kw || [];
  const now = ts.length ? new Date(ts[ts.length - 1]) : new Date();
  const carbonKgPerKwh = D.carbon_kg_per_kwh || 0.4;

  const fmtKwh = v => v >= 10000 ? (v / 1000).toFixed(1) + " MWh" : v.toFixed(0) + " kWh";
  const fmtDollars = v => v >= 10000 ? "$" + (v / 1000).toFixed(1) + "k" : "$" + v.toFixed(0);
  const fmtPct = (a, b) => { if (!b) return "\u2014"; const d = ((a - b) / b * 100); return (d >= 0 ? "+" : "") + d.toFixed(1) + "%"; };
  const trendClass = (a, b) => a > b ? "trend-up" : a < b ? "trend-down" : "trend-flat";
  const fmtMonth = d => d.toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
  const fmtTonnes = v => v >= 100 ? v.toFixed(0) + " tCO\u2082e" : v.toFixed(1) + " tCO\u2082e";
  const fmtKw = v => v >= 1000 ? (v / 1000).toFixed(2) + " MW" : v.toFixed(0) + " kW";

  function windowKwh(series, from, to) {
    let kwh = 0;
    for (let i = 1; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      if (ti < from || ti > to) continue;
      const h = Math.max(0, (ti - new Date(ts[i - 1])) / 3600000);
      kwh += (series[i] ?? 0) * h;
    }
    return kwh;
  }
  function windowAvgKw(series, from, to) {
    let sum = 0, cnt = 0;
    for (let i = 0; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      if (ti < from || ti > to) continue;
      sum += series[i] ?? 0;
      cnt++;
    }
    return cnt ? sum / cnt : 0;
  }
  function windowPeakKw(series, from, to) {
    let peak = 0;
    for (let i = 0; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      if (ti < from || ti > to) continue;
      peak = Math.max(peak, series[i] ?? 0);
    }
    return peak;
  }

  const mtdStart = new Date(now); mtdStart.setUTCDate(1); mtdStart.setUTCHours(0, 0, 0, 0);
  const prevMonthEnd = new Date(mtdStart); prevMonthEnd.setUTCMilliseconds(-1);
  const prevMonthStart = new Date(prevMonthEnd); prevMonthStart.setUTCDate(1); prevMonthStart.setUTCHours(0, 0, 0, 0);

  const mtdKwh = windowKwh(totalKw, mtdStart, now);
  const prevMonthKwh = windowKwh(totalKw, prevMonthStart, prevMonthEnd);
  const mtdCost = mtdKwh * PRICE;
  const prevMonthCost = prevMonthKwh * PRICE;
  const mtdCarbonTonnes = (mtdKwh * carbonKgPerKwh) / 1000;
  const prevMonthCarbonTonnes = (prevMonthKwh * carbonKgPerKwh) / 1000;
  const mtdPeak = windowPeakKw(totalKw, mtdStart, now);
  const prevPeak = windowPeakKw(totalKw, prevMonthStart, prevMonthEnd);
  const mtdAvg = windowAvgKw(totalKw, mtdStart, now);
  const prevAvg = windowAvgKw(totalKw, prevMonthStart, prevMonthEnd);
  const mtdLf = mtdPeak > 0 ? (mtdAvg / mtdPeak * 100) : 0;
  const prevLf = prevPeak > 0 ? (prevAvg / prevPeak * 100) : 0;

  const topCardsHtml = `
    <div class="exec-cards-grid cols-4">
      <div class="exec-card exec-headline">
        <div class="exec-section-label">Executive Snapshot <span class="exec-date-range">${fmtMonth(mtdStart)}</span></div>
        <div class="exec-headline-cost">${fmtDollars(mtdCost)}</div>
        <div class="exec-headline-val">${fmtKwh(mtdKwh)}</div>
        <div class="exec-headline-sub"><span class="${trendClass(mtdCost, prevMonthCost)}">${fmtPct(mtdCost, prevMonthCost)}</span> vs ${fmtMonth(prevMonthStart)} cost</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Carbon Impact (MTD)</div>
        <div class="exec-kpi-cost">${fmtTonnes(mtdCarbonTonnes)}</div>
        <div class="exec-kpi-sub"><span class="${trendClass(mtdCarbonTonnes, prevMonthCarbonTonnes)}">${fmtPct(mtdCarbonTonnes, prevMonthCarbonTonnes)}</span> vs ${fmtMonth(prevMonthStart)}</div>
        <div class="exec-kpi-dates">${carbonKgPerKwh.toFixed(3)} kg CO\u2082e / kWh</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Peak Demand (MTD)</div>
        <div class="exec-kpi-cost">${fmtKw(mtdPeak)}</div>
        <div class="exec-kpi-sub"><span class="${trendClass(mtdPeak, prevPeak)}">${fmtPct(mtdPeak, prevPeak)}</span> vs ${fmtMonth(prevMonthStart)}</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Load Factor (MTD)</div>
        <div class="exec-kpi-cost">${mtdLf.toFixed(1)}%</div>
        <div class="exec-kpi-sub"><span class="${trendClass(mtdLf, prevLf)}">${fmtPct(mtdLf, prevLf)}</span> vs ${fmtMonth(prevMonthStart)}</div>
      </div>
    </div>`;

  document.getElementById("execv-cards-top").innerHTML = topCardsHtml;
  document.getElementById("execv-cards-bottom").innerHTML = `
    <div class="exec-cards-grid cols-3">
      <div class="exec-card">
        <div class="exec-section-label">Storyline</div>
        <div style="color:var(--heading);font-size:0.85rem;line-height:1.5">
          Month-to-date spend is <strong>${fmtDollars(mtdCost)}</strong>, with carbon impact at <strong>${fmtTonnes(mtdCarbonTonnes)}</strong>.
          Peak demand is ${fmtKw(mtdPeak)}, and utilization quality (load factor) is ${mtdLf.toFixed(1)}%.
        </div>
      </div>
      <div class="exec-card">
        <div class="exec-section-label">Carbon Assumption</div>
        <div style="color:var(--heading);font-size:1.3rem;font-weight:700">${carbonKgPerKwh.toFixed(3)} kg CO\u2082e / kWh</div>
        <div style="color:var(--muted);font-size:0.75rem;margin-top:6px">Editable in <code>visualization_config.json</code> via <code>carbon_kg_per_kwh</code>.</div>
      </div>
      <div class="exec-card">
        <div class="exec-section-label">Estimated Savings Opportunity</div>
        <div style="color:var(--heading);font-size:1.3rem;font-weight:700">${fmtDollars(Math.max(0, prevMonthCost - mtdCost))}</div>
        <div style="color:var(--muted);font-size:0.75rem;margin-top:6px">Gap vs prior month total spend (directional).</div>
      </div>
    </div>`;

  const monthly = {};
  for (let i = 1; i < ts.length; i++) {
    const t = new Date(ts[i]);
    const prev = new Date(ts[i - 1]);
    const key = t.toISOString().slice(0, 7);
    const h = Math.max(0, (t - prev) / 3600000);
    const kwh = (totalKw[i] ?? 0) * h;
    if (!monthly[key]) monthly[key] = { kwh: 0 };
    monthly[key].kwh += kwh;
  }
  const months = Object.keys(monthly).sort();
  const costVals = months.map(m => monthly[m].kwh * PRICE);
  const carbonVals = months.map(m => monthly[m].kwh * carbonKgPerKwh / 1000);

  Plotly.newPlot("chart-execv-monthly", [
    { x: months, y: costVals, type: "bar", name: "Cost ($)", marker: { color: "#C41230" }, yaxis: "y1" },
    { x: months, y: carbonVals, type: "scatter", mode: "lines+markers", name: "Carbon (tCO\u2082e)", line: { color: "#2D3842", width: 2 }, yaxis: "y2" },
  ], pLayout({
    xaxis: xA({ type: "category" }),
    yaxis: yA({ title: "Cost ($)" }),
    yaxis2: Object.assign({}, yA({ title: "Carbon (tCO\u2082e)", overlaying: "y", side: "right" })),
    margin: { t: 12, l: 55, r: 55, b: 45 },
    legend: { orientation: "h", x: 0, y: 1.15 },
  }), pCfg);
}

async function _renderExecFloorPlan(panelSeries, groupSeries, groupNames, ts, now, fmtKwh, fmtDollars, fmtPct, trendClass, t, windowKwh, windowAvgKw, windowPeakKw, lastWeekStart, lastWeekEnd, lastSat, lastSunEnd, prevSat, prevSunEnd) {
  const col = document.getElementById("exec-floor-col");
  let plans = [];
  try { plans = await API.getFloorPlans(); } catch(e) {}
  const dashPlans = plans.filter(fp => fp.show_on_dashboard);

  // Time boundaries
  const weekAgo = new Date(now); weekAgo.setDate(weekAgo.getDate() - 7);
  const monthAgo = new Date(now); monthAgo.setMonth(monthAgo.getMonth() - 1);

  if (!dashPlans.length) {
    col.innerHTML = '<div class="panel-title">Facility Load Profile</div><div id="exec-fallback-load" style="height:100%;min-height:300px"></div>';
    const rolling = rollingMean(ts, D.total_kw || [], D.rolling_hours);
    setTimeout(() => {
      Plotly.newPlot("exec-fallback-load", [{
        x: ts, y: rolling, mode: "lines",
        line: { color: t.accent, width: 2.5, shape: "spline" },
        fill: "tozeroy", fillcolor: "rgba(196,18,48,0.08)",
        hovertemplate: "%{x}<br>%{y:.1f} kW<extra></extra>"
      }], pLayout({
        xaxis: xA({ type: "date" }), yaxis: yA({}),
        margin: { t: 8, l: 45, r: 15, b: 35 }
      }), pCfg);
    }, 0);
    return;
  }

  // Fetch all plan details in parallel
  const planDetails = await Promise.all(dashPlans.map(fp => API.getFloorPlan(fp.id)));
  const showTabs = dashPlans.length > 1;

  // Heatmap color scale: blue→green→yellow→orange→red
  function heatColor(ratio) {
    const r = Math.min(1, Math.max(0, ratio));
    const stops = [
      [0, 33, 150, 243],
      [0.25, 76, 175, 80],
      [0.5, 255, 193, 7],
      [0.75, 255, 152, 0],
      [1, 244, 67, 54],
    ];
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (r >= stops[i][0] && r <= stops[i+1][0]) { lo = stops[i]; hi = stops[i+1]; break; }
    }
    const f = lo[0] === hi[0] ? 0 : (r - lo[0]) / (hi[0] - lo[0]);
    const R = Math.round(lo[1] + f * (hi[1] - lo[1]));
    const G = Math.round(lo[2] + f * (hi[2] - lo[2]));
    const B = Math.round(lo[3] + f * (hi[3] - lo[3]));
    return `rgb(${R},${G},${B})`;
  }

  const fmtDate = d => d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  const fmtRange = (a, b) => fmtDate(a) + " \u2013 " + fmtDate(b);

  // Compute per-plan metrics and HTML
  const planData = planDetails.map((plan, idx) => {
    const zones = plan.zones || [];
    const deviceMetrics = {};
    const allWeekKwh = [];
    zones.forEach(zone => {
      const series = panelSeries[zone.device_id] || [];
      const weekKwh = windowKwh(series, lastWeekStart, lastWeekEnd);
      const monthKwh = windowKwh(series, monthAgo, now);
      const avgKw = windowAvgKw(series, lastWeekStart, lastWeekEnd);
      const peakKw = windowPeakKw(series, lastWeekStart, lastWeekEnd);
      const prevWkStart = new Date(lastWeekStart); prevWkStart.setUTCDate(prevWkStart.getUTCDate() - 7);
      const prevWkEnd = new Date(lastWeekStart); prevWkEnd.setUTCMilliseconds(-1);
      const prevWkKwh = windowKwh(series, prevWkStart, prevWkEnd);
      const trendPct = prevWkKwh > 0 ? ((weekKwh - prevWkKwh) / prevWkKwh * 100) : 0;
      const alertUp = trendPct >= 5;
      const alertDown = trendPct <= -5;
      const wkndKwh = windowKwh(series, lastSat, lastSunEnd);
      const prevWkndKwh = windowKwh(series, prevSat, prevSunEnd);
      const weekCost = weekKwh * PRICE;
      const monthCost = monthKwh * PRICE;
      deviceMetrics[zone.device_id] = { peakKw, avgKw, weekKwh, monthKwh, trendPct, alertUp, alertDown, wkndKwh, prevWkndKwh, weekCost, monthCost };
      allWeekKwh.push(weekKwh);
    });
    const maxKwh = Math.max(...allWeekKwh, 1);
    return { plan, zones, deviceMetrics, maxKwh };
  });

  // Build tab bar
  const tabsHtml = showTabs ? `<div class="fp-tabs">${dashPlans.map((fp, i) =>
    `<button class="fp-tab${i === 0 ? ' active' : ''}" data-fp-idx="${i}">${fp.name}</button>`
  ).join('')}</div>` : '';

  // Build per-plan pane HTML
  const panesHtml = planData.map((pd, idx) => {
    const { plan, zones, deviceMetrics, maxKwh } = pd;
    const zonesHtml = zones.map((zone, zi) => {
      const pts = zone.points || [];
      if (pts.length < 3) return "";
      const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const dm = deviceMetrics[zone.device_id] || {};
      const dev = ALL_DEVICES.find(d => d.id === zone.device_id);
      const label = zone.label || (dev ? dev.display_name : zone.device_id);
      let alertHtml = "";
      if (dm.alertUp) alertHtml = `<span class="fp-zone-alert" title="+${dm.trendPct.toFixed(0)}% vs last week">&#9650;</span>`;
      else if (dm.alertDown) alertHtml = `<span class="fp-zone-alert-down" title="${dm.trendPct.toFixed(0)}% vs last week">&#9660;</span>`;
      const zoneW = maxX - minX, zoneH = maxY - minY;
      const zoneSz = Math.min(zoneW, zoneH);
      const fontSize = zoneSz < 8 ? 0.45 : zoneSz < 15 ? 0.55 : 0.62;
      const valSize = fontSize * 0.85;
      return `<div class="fp-zone-label" data-zone-idx="${zi}" style="left:${cx}%;top:${cy}%;max-width:${zoneW * 0.9}%">
        <div class="fp-zone-label-text" style="font-size:${fontSize}rem">${alertHtml}${label}</div>
        <div class="fp-zone-label-val" style="font-size:${valSize}rem">${fmtDollars(dm.weekCost||0)}/wk</div>
        <div class="fp-dash-tooltip">
          <div class="fp-tt-title">${label}${dm.alertUp ? ' <span class="fp-zone-alert" style="font-size:0.7rem">&#9650; +' + dm.trendPct.toFixed(0) + '%</span>' : dm.alertDown ? ' <span class="fp-zone-alert-down" style="font-size:0.7rem">&#9660; ' + dm.trendPct.toFixed(0) + '%</span>' : ""}</div>
          <div class="fp-tt-row"><span>Last Full Week</span> <strong>${fmtKwh(dm.weekKwh||0)}</strong></div>
          <div class="fp-tt-row"><span>Weekly Cost</span> <strong class="fp-tt-cost">${fmtDollars(dm.weekCost||0)}</strong></div>
          <hr class="fp-tt-divider"/>
          <div class="fp-tt-row"><span>Last Weekend</span> <strong>${fmtKwh(dm.wkndKwh||0)} (${fmtDollars((dm.wkndKwh||0)*PRICE)})</strong></div>
          <div class="fp-tt-row"><span>Prev Weekend</span> <strong>${fmtKwh(dm.prevWkndKwh||0)} (${fmtDollars((dm.prevWkndKwh||0)*PRICE)})</strong></div>
          <hr class="fp-tt-divider"/>
          <div class="fp-tt-row"><span>Last 30 Days</span> <strong>${fmtKwh(dm.monthKwh||0)} (${fmtDollars(dm.monthCost||0)})</strong></div>
          <div class="fp-tt-row"><span>Week over Week</span> <strong class="${trendClass(dm.trendPct, 0)}">${dm.trendPct >= 0 ? "+" : ""}${(dm.trendPct||0).toFixed(0)}%</strong></div>
          <div class="fp-tt-row"><span>Avg Load</span> <strong>${(dm.avgKw||0).toFixed(1)} kW</strong></div>
          <div class="fp-tt-row"><span>Peak</span> <strong>${(dm.peakKw||0).toFixed(1)} kW</strong></div>
        </div>
      </div>`;
    }).join("");

    return `<div class="fp-tab-pane${idx === 0 ? '' : ' hidden'}" data-fp-idx="${idx}">
      <div class="fp-image-wrap" id="fp-wrap-${idx}">
        <img src="${plan.image_path}" id="fp-img-${idx}"/>
        <svg class="fp-heatmap-svg" id="fp-heatmap-svg-${idx}" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
        ${zonesHtml}
      </div>
    </div>`;
  }).join("");

  col.innerHTML = `
    <div class="panel-title" style="margin-bottom:0.5rem">${showTabs ? 'Floor Plans' : dashPlans[0].name}
      <span class="fp-heatmap-legend">
        <span class="fp-legend-low">Low</span>
        <span class="fp-legend-bar"></span>
        <span class="fp-legend-high">High</span>
      </span>
    </div>
    ${tabsHtml}
    ${panesHtml}`;

  // Draw heatmap polygons on each plan's SVG
  planData.forEach((pd, idx) => {
    const svg = document.getElementById("fp-heatmap-svg-" + idx);
    if (!svg) return;
    pd.zones.forEach((zone, zi) => {
      const pts = zone.points || [];
      if (pts.length < 3) return;
      const dm = pd.deviceMetrics[zone.device_id] || {};
      const ratio = pd.maxKwh > 0 ? (dm.weekKwh || 0) / pd.maxKwh : 0;
      const color = heatColor(ratio);
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("points", pts.map(p => `${p.x},${p.y}`).join(" "));
      poly.setAttribute("fill", color);
      poly.setAttribute("fill-opacity", "0.4");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "0.3");
      poly.setAttribute("stroke-opacity", "0.9");
      poly.dataset.zoneIdx = zi;
      poly.style.cursor = "pointer";
      poly.style.pointerEvents = "all";
      svg.appendChild(poly);
    });
  });

  // Wire zone hover tooltips for each plan
  planData.forEach((pd, idx) => {
    const wrap = document.getElementById("fp-wrap-" + idx);
    if (!wrap) return;
    wrap.querySelectorAll(".fp-zone-label").forEach(labelEl => {
      const tooltip = labelEl.querySelector(".fp-dash-tooltip");
      labelEl.addEventListener("mouseenter", () => {
        const wrapRect = wrap.getBoundingClientRect();
        const elRect = labelEl.getBoundingClientRect();
        tooltip.style.left = "0px";
        tooltip.style.top = "0px";
        wrap.appendChild(tooltip);
        tooltip.classList.add("fp-tt-visible");
        const ttW = tooltip.offsetWidth;
        const ttH = tooltip.offsetHeight;
        const cx = elRect.left + elRect.width / 2 - wrapRect.left;
        const cy = elRect.top - wrapRect.top;
        let left = cx - ttW / 2;
        left = Math.max(4, Math.min(left, wrapRect.width - ttW - 4));
        let top = cy - ttH - 8;
        if (top < 4) top = cy + elRect.height + 8;
        top = Math.max(4, Math.min(top, wrapRect.height - ttH - 4));
        tooltip.style.left = left + "px";
        tooltip.style.top = top + "px";
      });
      labelEl.addEventListener("mouseleave", () => {
        tooltip.classList.remove("fp-tt-visible");
        labelEl.appendChild(tooltip);
      });
    });
  });

  // Wire tab switching
  if (showTabs) {
    col.querySelectorAll('.fp-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        col.querySelectorAll('.fp-tab').forEach(b => b.classList.remove('active'));
        col.querySelectorAll('.fp-tab-pane').forEach(p => p.classList.add('hidden'));
        btn.classList.add('active');
        col.querySelector('.fp-tab-pane[data-fp-idx="' + btn.dataset.fpIdx + '"]').classList.remove('hidden');
      });
    });
  }
}

function renderAnalytics() {
  const data = filterForTab("analytics");
  const st = tabState.analytics;
  const t=T();

  const bk={};
  data.timestamps.forEach((ts,i)=>{ const d=new Date(ts),dk=d.toISOString().slice(0,10),h=d.getUTCHours();
    if(!bk[dk])bk[dk]={}; if(!bk[dk][h])bk[dk][h]={s:0,c:0}; bk[dk][h].s+=data.totalKw[i]??0; bk[dk][h].c++; });
  const dates=Object.keys(bk).sort(), hours=Array.from({length:24},(_,i)=>i);
  const z=hours.map(h=>dates.map(d=>{ const b=(bk[d]||{})[h]; return b?b.s/b.c:null; }));
  Plotly.newPlot("chart-heatmap",[{
    x:dates,y:hours,z, type:"heatmap",
    colorscale:isDark?[[0,"#0D1117"],[0.3,"#132D42"],[0.6,"#1D5B83"],[0.8,"#8BD435"],[1,"#F4FDE8"]]
      :[[0,"#F8FAFB"],[0.3,"#E3E8EC"],[0.6,"#6C7784"],[0.8,"#1B4A6C"],[1,"#8BD435"]],
    zsmooth:"best",connectgaps:true,
    colorbar:{title:{text:"kW",font:{size:11}},thickness:15}
  }],pLayout({
    xaxis:xA({title:{text:"Date",font:{size:12}},type:"category"}),
    yaxis:Object.assign(yA({title:{text:"Hour",font:{size:12}}}),{autorange:"reversed",rangemode:undefined})
  }),pCfg);

  const hp=hourlyProfile(data.timestamps,data.totalKw);
  const hpTraces=[{
    x:hp.hours,y:hp.avgs, mode:"lines+markers",
    line:{color:t.accentDark,width:2.5,shape:"spline"},
    marker:{size:7,color:t.accentDark,line:{color:t.card,width:2}},
    fill:"tozeroy",fillcolor:t.accentDark+"12"
  }];
  const hpLayout={
    xaxis:xA({title:{text:"Hour (UTC)",font:{size:12}},dtick:2}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}})
  };
  if (hasWeatherOverlay()) {
    const wx = filterWeather(st.startDate ? st.startDate+"T00:00:00Z" : null, st.endDate ? st.endDate+"T23:59:59Z" : null);
    if (wx) { hpTraces.push(...weatherHourlyProfileTraces(wx)); Object.assign(hpLayout, weatherAxesLayout()); }
  }
  Plotly.newPlot("chart-hourly",hpTraces,pLayout(hpLayout),pCfg);

  const wp=weekdayProfile(data.timestamps,data.totalKw);
  Plotly.newPlot("chart-weekday",[{
    x:wp.days,y:wp.avgs, type:"bar", marker:{color:t.accent}
  }],pLayout({
    xaxis:xA({title:{text:"Day",font:{size:12}}}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}}), bargap:0.2
  }),pCfg);

  const de=dailyEnergy(data.timestamps,data.totalKw);
  const deTraces=[{
    x:de.dates,y:de.values, type:"bar", marker:{color:t.accentDark}
  }];
  const deLayout={
    xaxis:xA({title:{text:"Date",font:{size:12}},type:"category"}),
    yaxis:yA({title:{text:"kWh",font:{size:12}}}), bargap:0.15
  };
  if (hasWeatherOverlay()) {
    const wx = filterWeather(st.startDate ? st.startDate+"T00:00:00Z" : null, st.endDate ? st.endDate+"T23:59:59Z" : null);
    if (wx) { deTraces.push(...weatherDailyTraces(wx)); Object.assign(deLayout, weatherAxesLayout()); }
  }
  Plotly.newPlot("chart-daily-energy",deTraces,pLayout(deLayout),pCfg);

  // Department load trends
  if (DEPARTMENTS.length) {
    const deptTraces = DEPARTMENTS.map(dept => {
      const panels = (DEPT_DEVICE_MAP[dept.id] || []).filter(p => ALL_PANELS.includes(p));
      if (!panels.length) return null;
      const y = data.timestamps.map((_, i) => {
        let sum = 0; panels.forEach(p => { sum += (data.panelSeries[p] || [])[i] || 0; }); return sum;
      });
      return { x: data.timestamps, y, mode: "lines", name: dept.display_name, line: { width: 2.5, shape: "spline", color: dept.color } };
    }).filter(Boolean);
    if (deptTraces.length) {
      document.getElementById("dept-chart-panel").style.display = "";
      Plotly.newPlot("chart-dept-trends", deptTraces, pLayout({
        legend: { orientation: "h", y: -0.15 },
        xaxis: xA({ title: { text: "Time", font: { size: 12 } }, type: "date" }),
        yaxis: yA({ title: { text: "kW", font: { size: 12 } } })
      }), pCfg);
    } else {
      document.getElementById("dept-chart-panel").style.display = "none";
    }
  }

  const gNames=Object.keys(data.groupSeries).filter(g=>{ const def=(D.group_definitions||[]).find(d=>d.name===g); return def&&def.panels&&def.panels.length; });
  if(gNames.length) {
    document.getElementById("group-chart-panel").style.display="";
    Plotly.newPlot("chart-groups",gNames.map(n=>({
      x:data.timestamps,y:data.groupSeries[n], mode:"lines",name:n,line:{width:2.5,shape:"spline"}
    })),pLayout({
      legend:{orientation:"h",y:-0.15},
      xaxis:xA({title:{text:"Time",font:{size:12}},type:"date"}),
      yaxis:yA({title:{text:"kW",font:{size:12}}})
    }),pCfg);
  }

  const pShow = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  if(pShow.length) {
    const panelTraces=pShow.map(p=>({
      x:data.timestamps, y:data.panelSeries[p]||[], mode:"lines",name:p,line:{width:2,shape:"spline"}
    }));
    const panelLayout={
      legend:{orientation:"h",y:-0.15},
      xaxis:xA({title:{text:"Time",font:{size:12}},type:"date"}),
      yaxis:yA({title:{text:"kW",font:{size:12}}}),
      shapes:weekendShapes(data.timestamps)
    };
    if (hasWeatherOverlay()) {
      const wx = filterWeather(st.startDate ? st.startDate+"T00:00:00Z" : null, st.endDate ? st.endDate+"T23:59:59Z" : null);
      if (wx) { panelTraces.push(...weatherTimeSeriesTraces(wx)); Object.assign(panelLayout, weatherAxesLayout()); }
    }
    Plotly.newPlot("chart-panels",panelTraces,pLayout(panelLayout),pCfg);
  }
}

function renderComparison() {
  const st = tabState.comparison;
  const t=T();
  const active = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  const allTs = D.timestamps, allKw = [];
  for (let i = 0; i < allTs.length; i++) {
    let tot = 0; active.forEach(p => { tot += (D.panel_series[p]||[])[i]||0; }); allKw.push(tot);
  }

  const p1s=new Date(st.p1Start+"T00:00:00Z"), p1e=new Date(st.p1End+"T23:59:59Z");
  const p2s=new Date(st.p2Start+"T00:00:00Z"), p2e=new Date(st.p2End+"T23:59:59Z");
  if(isNaN(p1s)||isNaN(p1e)||isNaN(p2s)||isNaN(p2e)) return;

  // Prominent period date headers
  const fmtD = d => d.toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric",timeZone:"UTC"});
  document.getElementById("comp-period-headers").innerHTML =
    `<div class="comp-period-hdr"><div class="comp-period-dot" style="background:${t.accent}"></div><div><div class="comp-period-hdr-label">Period 1</div><div class="comp-period-hdr-dates">${fmtD(p1s)} — ${fmtD(p1e)}</div></div></div>` +
    `<div class="comp-period-hdr"><div class="comp-period-dot" style="background:${t.accentDark}"></div><div><div class="comp-period-hdr-label">Period 2</div><div class="comp-period-hdr-dates">${fmtD(p2s)} — ${fmtD(p2e)}</div></div></div>`;

  const pd1={ts:[],kw:[]},pd2={ts:[],kw:[]};
  allTs.forEach((ts,i)=>{ const d=new Date(ts);
    if(d>=p1s&&d<=p1e){ pd1.ts.push(ts); pd1.kw.push(allKw[i]); }
    if(d>=p2s&&d<=p2e){ pd2.ts.push(ts); pd2.kw.push(allKw[i]); }
  });
  const m1=metrics(pd1.ts,pd1.kw), m2=metrics(pd2.ts,pd2.kw);
  const c1=m1.totalKwh*PRICE, c2=m2.totalKwh*PRICE;
  const lf1=m1.peakKw>0?m1.avgKw/m1.peakKw*100:0, lf2=m2.peakKw>0?m2.avgKw/m2.peakKw*100:0;
  const eSav=m1.totalKwh-m2.totalKwh, cSav=c1-c2, pRed=m1.peakKw-m2.peakKw, lfC=lf2-lf1;

  const ban=document.getElementById("sav-banner");
  if(eSav>0) {
    ban.className="savings-banner positive";
    document.getElementById("sav-title").textContent="Energy Savings Impact";
    document.getElementById("sav-energy").textContent=eSav.toFixed(0)+" kWh";
    document.getElementById("sav-cost").textContent="$"+cSav.toFixed(0);
  } else {
    ban.className="savings-banner negative";
    document.getElementById("sav-title").textContent="Energy Increase";
    document.getElementById("sav-energy").textContent=(-eSav).toFixed(0)+" kWh increase";
    document.getElementById("sav-cost").textContent="$"+(-cSav).toFixed(0)+" increase";
  }
  document.getElementById("sav-peak").textContent=pRed.toFixed(1)+" kW";
  document.getElementById("sav-lf").textContent=lfC.toFixed(1)+" pts";

  function setCC(id,cls) { document.getElementById("cc-"+id).className="comp-card "+cls; }

  // Update "Period 1" / "Period 2" labels in cards with short date ranges
  const fmtS = d => d.toLocaleDateString("en-US",{month:"short",day:"numeric",timeZone:"UTC"});
  const p1Label = `P1: ${fmtS(p1s)}–${fmtS(p1e)}`;
  const p2Label = `P2: ${fmtS(p2s)}–${fmtS(p2e)}`;
  document.querySelectorAll(".comp-card .comp-row").forEach((row, i) => {
    const span = row.querySelector(".comp-period");
    if (span) span.textContent = (i % 2 === 0) ? p1Label : p2Label;
  });

  document.getElementById("cc-e1").textContent=m1.totalKwh.toFixed(0)+" kWh";
  document.getElementById("cc-e2").textContent=m2.totalKwh.toFixed(0)+" kWh";
  const ced=document.getElementById("cc-ed"); ced.textContent=(m2.totalKwh-m1.totalKwh).toFixed(0)+" kWh";
  ced.className="delta-val "+(eSav>0?"pos":"neg"); setCC("energy",eSav>0?"savings":"increase");

  document.getElementById("cc-c1").textContent="$"+c1.toFixed(0);
  document.getElementById("cc-c2").textContent="$"+c2.toFixed(0);
  const ccd=document.getElementById("cc-cd"); ccd.textContent="$"+(c2-c1).toFixed(0);
  ccd.className="delta-val "+(cSav>0?"pos":"neg"); setCC("cost",cSav>0?"savings":"increase");

  document.getElementById("cc-a1").textContent=m1.avgKw.toFixed(1)+" kW";
  document.getElementById("cc-a2").textContent=m2.avgKw.toFixed(1)+" kW";
  const cad=document.getElementById("cc-ad"); cad.textContent=(m2.avgKw-m1.avgKw).toFixed(1)+" kW";
  cad.className="delta-val "+(m2.avgKw<m1.avgKw?"pos":"neg"); setCC("avg",m2.avgKw<m1.avgKw?"savings":"increase");

  document.getElementById("cc-p1").textContent=m1.peakKw.toFixed(1)+" kW";
  document.getElementById("cc-p2").textContent=m2.peakKw.toFixed(1)+" kW";
  const cpd=document.getElementById("cc-pd"); cpd.textContent=(m2.peakKw-m1.peakKw).toFixed(1)+" kW";
  cpd.className="delta-val "+(pRed>0?"pos":"neg"); setCC("peak",pRed>0?"savings":"increase");

  document.getElementById("cc-l1").textContent=lf1.toFixed(1)+"%";
  document.getElementById("cc-l2").textContent=lf2.toFixed(1)+"%";
  const cld=document.getElementById("cc-ld"); cld.textContent=lfC.toFixed(1)+" pts";
  cld.className="delta-val "+(lfC>0?"pos":"neg"); setCC("lf",lfC>0?"savings":"increase");

  const p1D=pd1.ts.length>0?Math.max(1,(new Date(pd1.ts[pd1.ts.length-1])-new Date(pd1.ts[0]))/86400000+1):1;
  const p2D=pd2.ts.length>0?Math.max(1,(new Date(pd2.ts[pd2.ts.length-1])-new Date(pd2.ts[0]))/86400000+1):1;
  document.getElementById("cc-d1").textContent=(m1.totalKwh/p1D).toFixed(0)+" kWh/d";
  document.getElementById("cc-d2").textContent=(m2.totalKwh/p2D).toFixed(0)+" kWh/d";
  const cdd=document.getElementById("cc-dd"); const dDiff=(m2.totalKwh/p2D)-(m1.totalKwh/p1D);
  cdd.textContent=dDiff.toFixed(0)+" kWh/d"; cdd.className="delta-val "+(dDiff<0?"pos":"neg");
  setCC("daily",dDiff<0?"savings":"increase");

  // Two side-by-side charts with same Y-axis range
  const yMax = Math.max(
    pd1.kw.length ? Math.max(...pd1.kw) : 0,
    pd2.kw.length ? Math.max(...pd2.kw) : 0
  ) * 1.08 || 1;
  const sharedY = yA({title:{text:"kW",font:{size:12}},range:[0,yMax]});

  const p1Traces=[
    {x:pd1.ts,y:pd1.kw,mode:"lines",name:"Period 1",line:{color:t.accent,width:2.5,shape:"spline"},fill:"tozeroy",fillcolor:t.accent+"18"}
  ];
  const p1Layout={xaxis:xA({title:{text:"",font:{size:11}},type:"date"}),yaxis:sharedY};
  const p2Traces=[
    {x:pd2.ts,y:pd2.kw,mode:"lines",name:"Period 2",line:{color:t.accentDark,width:2.5,shape:"spline"},fill:"tozeroy",fillcolor:t.accentDark+"18"}
  ];
  const p2Layout={xaxis:xA({title:{text:"",font:{size:11}},type:"date"}),yaxis:sharedY};
  if (hasWeatherOverlay()) {
    const wx1 = filterWeather(st.p1Start+"T00:00:00Z", st.p1End+"T23:59:59Z");
    if (wx1) { p1Traces.push(...weatherTimeSeriesTraces(wx1,{dash:"dot"})); Object.assign(p1Layout, weatherAxesLayout()); }
    const wx2 = filterWeather(st.p2Start+"T00:00:00Z", st.p2End+"T23:59:59Z");
    if (wx2) { p2Traces.push(...weatherTimeSeriesTraces(wx2,{dash:"dot"})); Object.assign(p2Layout, weatherAxesLayout()); }
  }
  Plotly.newPlot("chart-comp-load-p1",p1Traces,pLayout(p1Layout),pCfg);
  Plotly.newPlot("chart-comp-load-p2",p2Traces,pLayout(p2Layout),pCfg);

  const hp1=hourlyProfile(pd1.ts,pd1.kw),hp2=hourlyProfile(pd2.ts,pd2.kw);
  const compHourTraces=[
    {x:hp1.hours,y:hp1.avgs,mode:"lines+markers",name:"P1",line:{color:t.accent,width:2.5,shape:"spline"},marker:{size:7,color:t.accent}},
    {x:hp2.hours,y:hp2.avgs,mode:"lines+markers",name:"P2",line:{color:t.accentDark,width:2.5,shape:"spline"},marker:{size:7,color:t.accentDark}}
  ];
  const compHourLayout={ legend:{orientation:"h",y:-0.15},
    xaxis:xA({title:{text:"Hour",font:{size:12}},dtick:2}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}})
  };
  if (hasWeatherOverlay()) {
    const wxAll = filterWeather(
      new Date(Math.min(new Date(st.p1Start), new Date(st.p2Start))).toISOString(),
      new Date(Math.max(new Date(st.p1End+"T23:59:59Z"), new Date(st.p2End+"T23:59:59Z"))).toISOString()
    );
    if (wxAll) { compHourTraces.push(...weatherHourlyProfileTraces(wxAll)); Object.assign(compHourLayout, weatherAxesLayout()); }
  }
  Plotly.newPlot("chart-comp-hourly",compHourTraces,pLayout(compHourLayout),pCfg);

  const wp1=weekdayProfile(pd1.ts,pd1.kw),wp2=weekdayProfile(pd2.ts,pd2.kw);
  Plotly.newPlot("chart-comp-weekday",[
    {x:wp1.days,y:wp1.avgs,type:"bar",name:"P1",marker:{color:t.accent}},
    {x:wp2.days,y:wp2.avgs,type:"bar",name:"P2",marker:{color:t.accentDark}}
  ],pLayout({ legend:{orientation:"h",y:-0.15},
    xaxis:xA({title:{text:"Day",font:{size:12}}}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}}), barmode:"group"
  }),pCfg);
}

/* ═══════ DATA TABLE ═══════ */
let tData=[],tSortCol=0,tSortAsc=true,tPage=0;
const PAGE_SZ=50;

function renderDataTable() {
  const data = filterForTab("data");
  const st = tabState.data;
  const panels = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  tData=data.timestamps.map((ts,i)=>{ const r=[ts,(data.totalKw[i]??0).toFixed(2)];
    panels.forEach(p=>r.push((data.panelSeries[p]?.[i]??0).toFixed(2))); return r; });
  const thead=document.getElementById("table-head");
  thead.innerHTML=`<tr><th data-col="0">Timestamp <span class="sort-icon">&#9650;</span></th>
    <th data-col="1">Total kW <span class="sort-icon"></span></th>
    ${panels.map((p,i)=>`<th data-col="${i+2}">${p} <span class="sort-icon"></span></th>`).join("")}</tr>`;
  thead.querySelectorAll("th").forEach(th=>{
    th.addEventListener("click",()=>{ const c=parseInt(th.dataset.col);
      if(tSortCol===c)tSortAsc=!tSortAsc; else{tSortCol=c;tSortAsc=true;} renderTableRows(); });
  });
  tSortCol=0;tSortAsc=true;tPage=0; renderTableRows();
}

function renderTableRows() {
  const q=document.getElementById("table-search").value.toLowerCase();
  let f=tData; if(q) f=tData.filter(r=>r.some(c=>c.toLowerCase().includes(q)));
  f.sort((a,b)=>{ let va=a[tSortCol],vb=b[tSortCol];
    if(tSortCol>0){va=parseFloat(va);vb=parseFloat(vb);}
    if(va<vb)return tSortAsc?-1:1; if(va>vb)return tSortAsc?1:-1; return 0; });
  const tv=f.map(r=>parseFloat(r[1])), mean=tv.reduce((a,b)=>a+b,0)/(tv.length||1);
  const std=Math.sqrt(tv.reduce((a,b)=>a+(b-mean)**2,0)/(tv.length||1));
  const anomT=mean+2*std;
  const tp=Math.ceil(f.length/PAGE_SZ); tPage=Math.min(tPage,Math.max(0,tp-1));
  const s=tPage*PAGE_SZ, pg=f.slice(s,s+PAGE_SZ);
  document.getElementById("table-body").innerHTML=pg.map(r=>{
    const isA=parseFloat(r[1])>anomT;
    return `<tr${isA?' class="anomaly"':''}>${r.map(c=>`<td>${c}</td>`).join("")}</tr>`;
  }).join("");
  document.getElementById("table-info").textContent=`Showing ${s+1}-${Math.min(s+PAGE_SZ,f.length)} of ${f.length}`;
  const pag=document.getElementById("pagination"); pag.innerHTML="";
  if(tPage>0){ const b=document.createElement("button"); b.textContent="Prev"; b.addEventListener("click",()=>{tPage--;renderTableRows();}); pag.appendChild(b); }
  const sP=Math.max(0,tPage-3),eP=Math.min(tp,sP+7);
  for(let p=sP;p<eP;p++){ const b=document.createElement("button"); b.textContent=p+1; if(p===tPage)b.className="active";
    b.addEventListener("click",()=>{tPage=p;renderTableRows();}); pag.appendChild(b); }
  if(tPage<tp-1){ const b=document.createElement("button"); b.textContent="Next"; b.addEventListener("click",()=>{tPage++;renderTableRows();}); pag.appendChild(b); }
  document.querySelectorAll("#table-head th").forEach(th=>{ const ic=th.querySelector(".sort-icon"), c=parseInt(th.dataset.col);
    ic.innerHTML=c===tSortCol?(tSortAsc?"&#9650;":"&#9660;"):""; });
}

/* ═══════════════════════════════════════════════════════
   TAB DISPATCH
   ═══════════════════════════════════════════════════════ */
function renderTab(tabKey) {
  if      (tabKey === "executive")  renderExecutive();
  else if (tabKey === "overview")   renderOverview();
  else if (tabKey === "analytics")  renderAnalytics();
  else if (tabKey === "comparison") renderComparison();
  else if (tabKey === "data")       renderDataTable();
  else if (tabKey === "devices")    renderDevicesTab();
  else if (tabKey === "alerts")     renderAlertsTab();
  else if (tabKey === "health")     renderHealthTab();
  else if (tabKey === "behavior")   renderBehaviorTab();
}


async function renderAlertsTab() {
  const tableWrap = document.getElementById("alerts-table-wrap");
  const cards = document.getElementById("alerts-summary-cards");
  if (!tableWrap || !cards) return;

  await refreshAlerts(false);

  const includeAckEl = document.getElementById("alerts-include-ack");
  const windowEl = document.getElementById("alerts-window");
  if (includeAckEl) includeAckEl.checked = ALERTS_INCLUDE_ACK;
  if (windowEl) windowEl.value = String(ALERTS_WINDOW_HOURS);

  const refreshBtn = document.getElementById("alerts-refresh");
  const ackAllBtn = document.getElementById("alerts-ack-all");
  if (refreshBtn && !refreshBtn.dataset.wired) {
    refreshBtn.dataset.wired = "1";
    refreshBtn.addEventListener("click", async () => {
      ALERTS_INCLUDE_ACK = !!document.getElementById("alerts-include-ack")?.checked;
      ALERTS_WINDOW_HOURS = parseInt(document.getElementById("alerts-window")?.value || "24", 10);
      await refreshAlerts(false);
      renderAlertsTab();
    });
  }
  if (includeAckEl && !includeAckEl.dataset.wired) {
    includeAckEl.dataset.wired = "1";
    includeAckEl.addEventListener("change", async () => {
      ALERTS_INCLUDE_ACK = includeAckEl.checked;
      await refreshAlerts(false);
      renderAlertsTab();
    });
  }
  if (windowEl && !windowEl.dataset.wired) {
    windowEl.dataset.wired = "1";
    windowEl.addEventListener("change", async () => {
      ALERTS_WINDOW_HOURS = parseInt(windowEl.value || "24", 10);
      await refreshAlerts(false);
      renderAlertsTab();
    });
  }
  if (ackAllBtn && !ackAllBtn.dataset.wired) {
    ackAllBtn.dataset.wired = "1";
    ackAllBtn.addEventListener("click", async () => {
      const keys = ACTIVE_ALERTS.filter(a => !a.acknowledged).map(a => a.key);
      if (!keys.length) return;
      await API.acknowledgeAlerts(keys);
      await refreshAlerts(false);
      renderAlertsTab();
    });
  }

  const crit = ACTIVE_ALERTS.filter(a => a.severity === "critical").length;
  const warn = ACTIVE_ALERTS.filter(a => a.severity === "warning").length;
  const unacked = ACTIVE_ALERTS.filter(a => !a.acknowledged).length;
  cards.innerHTML = `<div class="exec-cards-grid cols-4">
    <div class="exec-card"><div class="exec-metric-label">Critical Alerts</div><div class="exec-metric-value">${crit}</div></div>
    <div class="exec-card"><div class="exec-metric-label">Warning Alerts</div><div class="exec-metric-value">${warn}</div></div>
    <div class="exec-card"><div class="exec-metric-label">Unacknowledged</div><div class="exec-metric-value">${unacked}</div></div>
    <div class="exec-card"><div class="exec-metric-label">Threshold-configured Devices</div><div class="exec-metric-value">${ALERT_SUMMARY.configured_devices || 0}</div></div>
  </div>`;

  const rows = ACTIVE_ALERTS.map(a => `<tr>
    <td>${a.timestamp}</td>
    <td>${a.device_name || a.device_id}</td>
    <td><span class="badge ${a.severity === "critical" ? "badge-neg" : "badge-warn"}">${a.severity}</span></td>
    <td>${a.current_kw.toFixed(2)}</td>
    <td>${a.threshold_kw == null ? "-" : Number(a.threshold_kw).toFixed(2)}</td>
    <td>${a.acknowledged ? '<span class="kpi-badge positive">Acknowledged</span>' : '<span class="kpi-badge negative">Active</span>'}</td>
    <td>${a.acknowledged ? "" : `<button class="btn btn-ghost btn-sm ack-row" data-key="${a.key}">Acknowledge</button>`}</td>
  </tr>`).join("");

  tableWrap.innerHTML = `<table class="data-table"><thead><tr>
    <th>Timestamp</th><th>Device</th><th>Severity</th><th>Current kW</th><th>Threshold kW</th><th>Status</th><th>Action</th>
  </tr></thead><tbody>${rows || '<tr><td colspan="7">No alerts for selected window.</td></tr>'}</tbody></table>`;

  tableWrap.querySelectorAll(".ack-row").forEach(btn => {
    btn.addEventListener("click", async () => {
      await API.acknowledgeAlerts([btn.dataset.key]);
      await refreshAlerts(false);
      renderAlertsTab();
    });
  });
}

async function renderHealthTab() {
  const cards = document.getElementById("health-summary-cards");
  const failures = document.getElementById("health-failures");
  const freshness = document.getElementById("health-freshness");
  if (!cards || !failures || !freshness) return;
  const payload = await API.getDataHealth();
  const s = payload.summary || {};

  cards.innerHTML = `<div class="exec-cards-grid cols-4">
    <div class="exec-card"><div class="exec-metric-label">Ingest Events</div><div class="exec-metric-value">${s.total_ingest_events || 0}</div></div>
    <div class="exec-card"><div class="exec-metric-label">Failed Events</div><div class="exec-metric-value">${s.failed_events || 0}</div></div>
    <div class="exec-card"><div class="exec-metric-label">Success Rate</div><div class="exec-metric-value">${(s.success_rate_pct || 0).toFixed(1)}%</div></div>
    <div class="exec-card"><div class="exec-metric-label">Data Staleness</div><div class="exec-metric-value">${s.is_data_stale ? "Stale" : "Fresh"}</div></div>
  </div>`;

  const failRows = (payload.recent_failures || []).slice(0, 12).map(f =>
    `<tr><td>${f.processed_at || "-"}</td><td>${f.filename || "-"}</td><td>${f.error_message || "-"}</td></tr>`
  ).join("");
  failures.innerHTML = `<div class="data-table-wrap"><table class="data-table"><thead><tr><th>Time</th><th>File</th><th>Error</th></tr></thead><tbody>${failRows || '<tr><td colspan="3">No recent failures</td></tr>'}</tbody></table></div>`;

  const freshRows = (payload.device_freshness || []).slice(0, 20).map(d =>
    `<tr><td>${d.device_id}</td><td>${d.last_seen || "Never"}</td><td>${d.latest_value == null ? "-" : d.latest_value.toFixed(2)}</td></tr>`
  ).join("");
  freshness.innerHTML = `<div class="data-table-wrap"><table class="data-table"><thead><tr><th>Device</th><th>Last Seen</th><th>Latest Amps</th></tr></thead><tbody>${freshRows || '<tr><td colspan="3">No devices</td></tr>'}</tbody></table></div>`;
}

function renderCurrentTab() {
  renderTab(activeTab);
}

/* ═══════════════════════════════════════════════════════
   BEHAVIOR TAB
   ═══════════════════════════════════════════════════════ */
function buildBehaviorFilterBar() {
  const container = document.getElementById("bh-filter-bar");
  if (!container) return;
  const st = tabState.behavior;

  let html = '';
  html += '<div class="filter-group"><label>Panel / Device</label>';
  html += '<select id="bh-panel" class="filter-input" style="min-width:220px">';
  ALL_PANELS.forEach(p => {
    const dev = ALL_DEVICES.find(d => d.id === p);
    const label = dev ? dev.display_name + " (" + p + ")" : p;
    html += '<option value="' + p + '"' + (p === st.panel ? ' selected' : '') + '>' + label + '</option>';
  });
  html += '</select></div>';

  html += '<div class="filter-divider"></div>';
  html += '<div class="filter-group"><label>Start</label>';
  html += '<input type="date" id="bh-start" class="filter-input" value="' + st.startDate + '" min="' + DATE_MIN + '" max="' + DATE_MAX + '"/></div>';
  html += '<div class="filter-group"><label>End</label>';
  html += '<input type="date" id="bh-end" class="filter-input" value="' + st.endDate + '" min="' + DATE_MIN + '" max="' + DATE_MAX + '"/></div>';
  html += '<button class="btn btn-primary btn-sm" id="bh-analyze">Analyze</button>';

  container.innerHTML = html;

  document.getElementById("bh-analyze").addEventListener("click", () => {
    tabState.behavior.panel = document.getElementById("bh-panel").value;
    tabState.behavior.startDate = document.getElementById("bh-start").value;
    tabState.behavior.endDate = document.getElementById("bh-end").value;
    renderBehaviorTab();
  });
}

async function renderBehaviorTab() {
  const st = tabState.behavior;
  const t = T();
  const narrative = document.getElementById("bh-narrative");
  const cards = document.getElementById("bh-kpi-cards");
  const detail = document.getElementById("bh-narrative-detail");

  if (!st.panel) {
    if (narrative) narrative.innerHTML =
      '<div style="text-align:center;color:var(--muted);padding:3rem">' +
      '<h2>Select a Panel</h2><p>Choose a panel or device above to analyze its power usage behavior.</p></div>';
    return;
  }

  if (narrative) narrative.innerHTML =
    '<div style="text-align:center;color:var(--muted);padding:2rem">Analyzing behavior\u2026</div>';

  try {
    const bh = await API.getBehavior({
      panel: st.panel,
      start: st.startDate ? st.startDate + "T00:00:00Z" : undefined,
      end: st.endDate ? st.endDate + "T23:59:59Z" : undefined,
    });

    renderBehaviorNarrative(bh, narrative);
    renderBehaviorKPIs(bh, cards);
    renderBehaviorHourlyChart(bh, t);
    renderBehaviorSplitChart(bh, t);
    renderBehaviorScenariosChart(bh, t);
    renderBehaviorDetailNarrative(bh, detail);

  } catch (err) {
    console.error("Behavior analysis failed:", err);
    if (narrative) narrative.innerHTML =
      '<div style="text-align:center;color:var(--negative-text,#dc2626);padding:2rem">' +
      '<h3>Analysis Failed</h3><p>' + err.message + '</p></div>';
  }
}

function renderBehaviorNarrative(bh, container) {
  if (!container) return;
  const n = bh.narrative;
  const cls = bh.energy_split.off_shift_pct >= 40 ? "negative" : "positive";
  container.innerHTML =
    '<div class="savings-banner ' + cls + '" style="margin-bottom:var(--card-gap)">' +
    '<h3 style="font-size:1.4rem;margin-bottom:8px">' + n.headline + '</h3>' +
    '<p style="opacity:0.9;font-size:0.95rem;line-height:1.7;max-width:72ch">' + n.phantom_story + '</p>' +
    '</div>';
}

function renderBehaviorKPIs(bh, container) {
  if (!container) return;
  const es = bh.energy_split;
  const ph = bh.phantom_draw;
  const an = bh.annualized_phantom;

  container.innerHTML =
    '<div class="exec-cards-grid cols-4" style="margin-bottom:var(--card-gap)">' +
      '<div class="exec-card">' +
        '<div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.03em">Off-Shift Energy</div>' +
        '<div style="font-size:1.5rem;font-weight:800;color:var(--p-400,#C41230)">' + es.off_shift_pct.toFixed(0) + '%</div>' +
        '<div style="font-size:0.75rem;color:var(--muted)">' + es.off_shift_kwh.toFixed(0) + ' kWh of ' + es.total_kwh.toFixed(0) + ' kWh</div>' +
      '</div>' +
      '<div class="exec-card">' +
        '<div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.03em">Phantom Draw</div>' +
        '<div style="font-size:1.5rem;font-weight:800;color:var(--ink)">' + ph.phantom_kw.toFixed(2) + ' kW</div>' +
        '<div style="font-size:0.75rem;color:var(--muted)">Sustained minimum load overnight</div>' +
      '</div>' +
      '<div class="exec-card">' +
        '<div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.03em">Annual Phantom Cost</div>' +
        '<div style="font-size:1.5rem;font-weight:800;color:var(--p-400,#C41230)">$' + an.annual_phantom_cost.toLocaleString() + '</div>' +
        '<div style="font-size:0.75rem;color:var(--muted)">' + an.annual_phantom_kwh.toLocaleString() + ' kWh/year wasted</div>' +
      '</div>' +
      '<div class="exec-card">' +
        '<div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.03em">Annual Carbon Impact</div>' +
        '<div style="font-size:1.5rem;font-weight:800;color:var(--ink)">' + an.annual_phantom_carbon_kg.toLocaleString() + ' kg CO\u2082</div>' +
        '<div style="font-size:0.75rem;color:var(--muted)">' + an.annual_phantom_carbon_tonnes.toFixed(1) + ' tonnes CO\u2082/year</div>' +
      '</div>' +
    '</div>';
}

function renderBehaviorHourlyChart(bh, t) {
  const hp = bh.hourly_profile;

  // Shift band shape (6:00-14:30)
  const shiftShape = {
    type: "rect", xref: "x", yref: "paper",
    x0: 6, x1: 14.5, y0: 0, y1: 1,
    fillcolor: "rgba(22,163,74,0.08)",
    line: { width: 0 },
  };

  const traces = [
    {
      x: hp.hours, y: hp.avg_kw,
      mode: "lines+markers", name: "Overall Avg kW",
      line: { color: t.accentDark, width: 3, shape: "spline" },
      marker: { size: 8, color: t.accentDark, line: { color: t.card, width: 2 } },
      fill: "tozeroy", fillcolor: t.accentDark + "12",
    },
    {
      x: hp.hours, y: hp.off_shift_avg_kw,
      mode: "lines", name: "Off-Shift Avg kW",
      line: { color: t.accent, width: 2, dash: "dash", shape: "spline" },
    },
  ];

  // Phantom draw reference line
  const phantomKw = bh.phantom_draw.phantom_kw;
  if (phantomKw > 0) {
    traces.push({
      x: [0, 23], y: [phantomKw, phantomKw],
      mode: "lines", name: "Phantom Draw (" + phantomKw.toFixed(2) + " kW)",
      line: { color: "#EF4444", width: 2, dash: "dot" },
    });
  }

  Plotly.newPlot("chart-bh-hourly", traces, pLayout({
    xaxis: xA({ title: { text: "Hour of Day (Eastern)", font: { size: 12 } }, dtick: 1, range: [-0.5, 23.5] }),
    yaxis: yA({ title: { text: "kW", font: { size: 12 } } }),
    shapes: [shiftShape],
    annotations: [{
      x: 10.25, y: 1, xref: "x", yref: "paper",
      text: "Shift (6 AM \u2013 2:30 PM)", showarrow: false,
      font: { size: 10, color: "rgba(22,163,74,0.6)" }, yanchor: "top",
    }],
    legend: { orientation: "h", y: -0.2 },
  }), pCfg);
}

function renderBehaviorSplitChart(bh, t) {
  const es = bh.energy_split;

  Plotly.newPlot("chart-bh-split", [{
    values: [es.shift_kwh, es.off_shift_kwh],
    labels: ["Shift (" + es.shift_kwh.toFixed(0) + " kWh)", "Off-Shift (" + es.off_shift_kwh.toFixed(0) + " kWh)"],
    type: "pie", hole: 0.55,
    marker: { colors: ["#16A34A", "#EF4444"] },
    textinfo: "percent",
    textfont: { size: 14, color: "#fff" },
    hovertemplate: "%{label}<br>%{value:.0f} kWh<br>%{percent}<extra></extra>",
  }], pLayout({
    annotations: [{
      text: es.off_shift_pct.toFixed(0) + "%<br>Off-Shift",
      font: { size: 18, color: t.ink, weight: 700 },
      showarrow: false, x: 0.5, y: 0.5,
    }],
    legend: { orientation: "h", y: -0.1 },
  }), pCfg);
}

function renderBehaviorScenariosChart(bh, t) {
  const sc = bh.reduction_scenarios;

  Plotly.newPlot("chart-bh-scenarios", [
    {
      x: sc.map(s => s.reduction_pct + "% Reduction"),
      y: sc.map(s => s.annual_cost_saved),
      type: "bar", name: "Annual $ Saved",
      marker: { color: ["#38BDF8", "#16A34A", "#F97316", t.accent] },
      text: sc.map(s => "$" + s.annual_cost_saved.toLocaleString()),
      textposition: "outside",
      textfont: { size: 12, color: t.ink },
    },
  ], pLayout({
    xaxis: xA({ title: { text: "Phantom Reduction Level", font: { size: 12 } } }),
    yaxis: yA({ title: { text: "Annual Savings ($)", font: { size: 12 } } }),
    bargap: 0.3,
  }), pCfg);
}

function renderBehaviorDetailNarrative(bh, container) {
  if (!container) return;
  const n = bh.narrative;
  const sc = bh.reduction_scenarios;
  const full = sc.find(s => s.reduction_pct === 100);

  let fullKpis = '';
  if (full) {
    fullKpis =
      '<div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--card-border,#e5e7eb)">' +
        '<div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;margin-bottom:10px">If Phantom Draw Were Eliminated Entirely</div>' +
        '<div class="exec-cards-grid cols-4">' +
          '<div class="exec-card"><div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase">kWh Saved / Year</div><div style="font-size:1.3rem;font-weight:800;color:var(--ink)">' + full.annual_kwh_saved.toLocaleString() + '</div></div>' +
          '<div class="exec-card"><div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase">Cost Saved / Year</div><div style="font-size:1.3rem;font-weight:800;color:var(--p-400,#C41230)">$' + full.annual_cost_saved.toLocaleString() + '</div></div>' +
          '<div class="exec-card"><div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase">Equivalent Trees</div><div style="font-size:1.3rem;font-weight:800;color:var(--ink)">' + full.equivalent_trees_year.toFixed(0) + ' trees</div></div>' +
          '<div class="exec-card"><div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase">Equivalent Miles Not Driven</div><div style="font-size:1.3rem;font-weight:800;color:var(--ink)">' + full.equivalent_miles_driven.toLocaleString() + '</div></div>' +
        '</div>' +
      '</div>';
  }

  container.innerHTML =
    '<div class="panel" style="margin-top:var(--card-gap)">' +
      '<div class="panel-title">Impact Analysis</div>' +
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--card-gap,1.5rem)">' +
        '<div>' +
          '<h4 style="color:var(--ink);font-size:1rem;margin-bottom:8px">The Cost of Inaction</h4>' +
          '<p style="color:var(--muted);line-height:1.7;font-size:0.9rem">' + n.cost_impact + '</p>' +
        '</div>' +
        '<div>' +
          '<h4 style="color:var(--ink);font-size:1rem;margin-bottom:8px">What You Can Do</h4>' +
          '<p style="color:var(--muted);line-height:1.7;font-size:0.9rem">' + n.call_to_action + '</p>' +
        '</div>' +
      '</div>' +
      fullKpis +
    '</div>';
}

/* ─── Boot ─── */
document.addEventListener("DOMContentLoaded", initDashboard);
