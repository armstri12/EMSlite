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
let activeTab = "overview";
let DEPARTMENTS = [];       // [{id, display_name, color, device_count}]
let DEPT_DEVICE_MAP = {};   // department_id -> [panel_id, ...]

/* ─── Theme Palettes ─── */
const lightC = {
  ink:"#3A424D", muted:"#9BA5B0", card:"#ffffff", bg:"#F8FAFB",
  grid:"rgba(0,0,0,0.06)", accent:"#8BD435", accentDark:"#0B1E2E",
  gradFrom:"#0B1E2E", gradTo:"#8BD435", donutA:"#8BD435", donutB:"#132D42",
  series:["#8BD435","#0B1E2E","#398EB9","#F97316","#EF4444","#38BDF8","#6DBF1A","#1D5B83"]
};
const darkC = {
  ink:"#E3E8EC", muted:"#6C7784", card:"#171C23", bg:"#0D1117",
  grid:"rgba(255,255,255,0.06)", accent:"#8BD435", accentDark:"#E3E8EC",
  gradFrom:"#132D42", gradTo:"#8BD435", donutA:"#8BD435", donutB:"#1B4A6C",
  series:["#8BD435","#38BDF8","#F97316","#EF4444","#74B6D5","#6DBF1A","#D4EAF2","#ABDF65"]
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

    // Build department state
    DEPARTMENTS = departments || [];
    DEPT_DEVICE_MAP = {};
    for (const dept of DEPARTMENTS) {
      DEPT_DEVICE_MAP[dept.id] = [];
    }
    for (const dev of (devices || [])) {
      if (dev.department_id && DEPT_DEVICE_MAP[dev.department_id]) {
        DEPT_DEVICE_MAP[dev.department_id].push(dev.id);
      }
    }

    initTabState();
    setupThemeToggle();
    setupSidebar();
    setupTabNavigation();
    // Overview has no filter bar — it auto-displays latest data
    buildFilterBar("an-filter-bar", "analytics",  "daterange");
    buildFilterBar("cp-filter-bar", "comparison", "comparison");
    buildFilterBar("dt-filter-bar", "data",       "daterange");
    setupTableControls();
    renderOverview();
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
  data:       { panels: new Set(), startDate: "", endDate: "", department: "" }
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

/* ═══════════════════════════════════════════════════════
   TAB RENDERERS
   ═══════════════════════════════════════════════════════ */

async function renderOverview() {
  const t = T();
  // Use ALL data — no filters on the executive dashboard
  const ts = D.timestamps || [];
  const totalKw = D.total_kw || [];
  const panelSeries = D.panel_series || {};
  const lastIdx = ts.length - 1;

  // Global facility metrics
  const m = metrics(ts, totalKw);
  const currentKw = lastIdx >= 0 ? (totalKw[lastIdx] || 0) : 0;
  const lf = m.peakKw > 0 ? currentKw / m.peakKw * 100 : 0;
  const now = ts.length ? new Date(ts[lastIdx]) : new Date();

  // Rolling windows: today, 7 days, 30 days
  const todayStart = new Date(now); todayStart.setHours(0,0,0,0);
  const weekAgo = new Date(now); weekAgo.setDate(weekAgo.getDate() - 7);
  const monthAgo = new Date(now); monthAgo.setMonth(monthAgo.getMonth() - 1);
  let todayKwh = 0, weekKwh = 0, monthKwh = 0;
  for (let i = 1; i < ts.length; i++) {
    const ti = new Date(ts[i]);
    const h = Math.max(0, (ti - new Date(ts[i-1])) / 3600000);
    const v = totalKw[i] ?? 0;
    if (ti >= todayStart) todayKwh += v * h;
    if (ti >= weekAgo) weekKwh += v * h;
    if (ti >= monthAgo) monthKwh += v * h;
  }
  const todayCost = todayKwh * PRICE;

  const fmtKwh = v => v >= 10000 ? (v/1000).toFixed(1) + " MWh" : v >= 100 ? v.toFixed(0) + " kWh" : v.toFixed(1) + " kWh";
  const fmtKw = v => v >= 1000 ? (v/1000).toFixed(1) + " MW" : v.toFixed(1) + " kW";

  // Gauge color
  const gaugeColor = lf >= 80 ? '#EF4444' : lf >= 50 ? '#F97316' : '#8BD435';

  // Department breakdown (compact)
  const deptRows = DEPARTMENTS.map(dept => {
    const panels = DEPT_DEVICE_MAP[dept.id] || [];
    let kw = 0;
    panels.forEach(p => { kw += lastIdx >= 0 ? ((panelSeries[p] || [])[lastIdx] || 0) : 0; });
    return { name: dept.display_name, color: dept.color || t.accent, kw, count: panels.length };
  }).filter(d => d.count > 0).sort((a,b) => b.kw - a.kw);
  const deptTotal = deptRows.reduce((s,d) => s + d.kw, 0) || 1;
  const deptMax = deptRows.length ? deptRows[0].kw : 1;

  // Top consumers by CURRENT kW (not cumulative)
  const topConsumers = ALL_PANELS.map(p => {
    const kw = lastIdx >= 0 ? ((panelSeries[p] || [])[lastIdx] || 0) : 0;
    return { name: p, kw };
  }).sort((a,b) => b.kw - a.kw).slice(0, 5);
  const topMax = topConsumers.length ? topConsumers[0].kw : 1;

  // ── Build stats sidebar HTML ──
  const statsHtml = `
    <div class="exec-gauge">
      <div class="exec-gauge-title">Live Facility Load</div>
      <div class="exec-gauge-value">${fmtKw(currentKw)}</div>
      <div class="exec-gauge-bar"><div class="exec-gauge-fill" style="width:${Math.min(lf,100).toFixed(0)}%;background:${gaugeColor}"></div></div>
      <div class="exec-gauge-sub">${lf.toFixed(0)}% of peak (${fmtKw(m.peakKw)})</div>
    </div>
    <div class="exec-kpi-grid">
      <div class="exec-kpi">
        <div class="exec-kpi-label">Today</div>
        <div class="exec-kpi-val">${fmtKwh(todayKwh)}</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Est. Cost Today</div>
        <div class="exec-kpi-val">$${todayCost >= 1000 ? (todayCost/1000).toFixed(1)+'k' : todayCost.toFixed(0)}</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Last 7 Days</div>
        <div class="exec-kpi-val">${fmtKwh(weekKwh)}</div>
      </div>
      <div class="exec-kpi">
        <div class="exec-kpi-label">Last 30 Days</div>
        <div class="exec-kpi-val">${fmtKwh(monthKwh)}</div>
      </div>
    </div>
    ${deptRows.length ? `<div class="exec-card">
      <div class="exec-section-label">Departments (Live)</div>
      ${deptRows.map(d => `<div class="exec-dept-row">
        <div class="exec-dept-dot" style="background:${d.color}"></div>
        <div class="exec-dept-name">${d.name}</div>
        <div class="exec-dept-bar-wrap"><div class="exec-dept-bar-fill" style="width:${(d.kw/deptMax*100).toFixed(0)}%;background:${d.color}"></div></div>
        <div class="exec-dept-pct">${(d.kw/deptTotal*100).toFixed(0)}%</div>
        <div class="exec-dept-val">${fmtKw(d.kw)}</div>
      </div>`).join("")}
    </div>` : ""}
    ${topConsumers.length ? `<div class="exec-card">
      <div class="exec-section-label">Top Consumers (Live)</div>
      ${topConsumers.map(c => `<div class="exec-top-row">
        <div class="exec-top-name">${c.name}</div>
        <div class="exec-top-bar-wrap"><div class="exec-top-bar-fill" style="width:${(c.kw/(topMax||1)*100).toFixed(0)}%"></div></div>
        <div class="exec-top-val">${fmtKw(c.kw)}</div>
      </div>`).join("")}
    </div>` : ""}`;

  document.getElementById("exec-stats-col").innerHTML = statsHtml;

  // ── Floor plan hero (left column) ──
  await _renderExecFloorPlan(panelSeries, ts, now, fmtKwh, fmtKw, t);

  // ── 24h load strip at the bottom ──
  _renderLoadStrip(ts, totalKw, t, m);
}

async function _renderExecFloorPlan(panelSeries, ts, now, fmtKwh, fmtKw, t) {
  const col = document.getElementById("exec-floor-col");
  let plans = [];
  try { plans = await API.getFloorPlans(); } catch(e) {}
  const dashPlans = plans.filter(fp => fp.show_on_dashboard);

  if (!dashPlans.length) {
    // Fallback: show load profile chart instead of floor plan
    col.innerHTML = '<div class="panel-title">Facility Load Profile</div><div id="exec-fallback-load" style="height:100%;min-height:300px"></div>';
    const rolling = rollingMean(ts, D.total_kw || [], D.rolling_hours);
    setTimeout(() => {
      Plotly.newPlot("exec-fallback-load", [{
        x: ts, y: rolling, mode: "lines",
        line: { color: t.accent, width: 2.5, shape: "spline" },
        fill: "tozeroy", fillcolor: "rgba(139,212,53,0.1)",
        hovertemplate: "%{x}<br>%{y:.1f} kW<extra></extra>"
      }], pLayout({
        xaxis: xA({ type: "date" }), yaxis: yA({}),
        margin: { t: 8, l: 45, r: 15, b: 35 }
      }), pCfg);
    }, 0);
    return;
  }

  // Render the first dashboard floor plan as hero
  const fp = dashPlans[0];
  const plan = await API.getFloorPlan(fp.id);
  const pins = plan.pins || [];
  const lastIdx = ts.length - 1;

  // Compute per-device metrics
  const deviceMetrics = {};
  pins.forEach(pin => {
    const series = panelSeries[pin.device_id] || [];
    const currentKw = lastIdx >= 0 ? (series[lastIdx] || 0) : 0;
    const m = metrics(ts, series);
    const weekAgo = new Date(now); weekAgo.setDate(weekAgo.getDate() - 7);
    const monthAgo = new Date(now); monthAgo.setMonth(monthAgo.getMonth() - 1);
    let weekKwh = 0, monthKwh = 0;
    for (let i = 1; i < ts.length; i++) {
      const ti = new Date(ts[i]);
      const v = series[i] ?? 0;
      const h = Math.max(0, (ti - new Date(ts[i-1])) / 3600000);
      if (ti >= weekAgo) weekKwh += v * h;
      if (ti >= monthAgo) monthKwh += v * h;
    }
    deviceMetrics[pin.device_id] = { currentKw, peakKw: m.peakKw, avgKw: m.avgKw, weekKwh, monthKwh, totalKwh: m.totalKwh };
  });

  col.innerHTML = `
    <div class="panel-title" style="margin-bottom:0.5rem">${fp.name}</div>
    <div class="fp-image-wrap" id="fp-wrap-hero">
      <img src="${plan.image_path}" id="fp-img-hero"/>
      ${pins.map((pin, idx) => {
        const dm = deviceMetrics[pin.device_id] || {};
        const label = pin.label || pin.device_id;
        const kw = (dm.currentKw || 0).toFixed(1);
        const ratio = dm.peakKw > 0 ? dm.currentKw / dm.peakKw : 0;
        const dotColor = ratio >= 0.8 ? '#EF4444' : ratio >= 0.5 ? '#F97316' : '#8BD435';
        return `<div class="fp-dash-pin" data-idx="${idx}" style="left:${pin.x_pct}%;top:${pin.y_pct}%">
          <div class="fp-pin-dot" style="background:${dotColor}"></div>
          <div class="fp-pin-tag">${label}: ${kw} kW</div>
          <div class="fp-dash-tooltip">
            <div class="fp-tt-title">${label}</div>
            <div class="fp-tt-row"><span>Current</span> <strong>${kw} kW</strong></div>
            <div class="fp-tt-row"><span>Peak</span> <strong>${(dm.peakKw||0).toFixed(1)} kW</strong></div>
            <div class="fp-tt-row"><span>Avg</span> <strong>${(dm.avgKw||0).toFixed(1)} kW</strong></div>
            <div class="fp-tt-row"><span>Last 7 days</span> <strong>${fmtKwh(dm.weekKwh||0)}</strong></div>
            <div class="fp-tt-row"><span>Last 30 days</span> <strong>${fmtKwh(dm.monthKwh||0)}</strong></div>
          </div>
        </div>`;
      }).join("")}
    </div>`;

  const wrap = document.getElementById("fp-wrap-hero");

  // ── Tooltip: JS show/hide with smart positioning ──
  wrap.querySelectorAll(".fp-dash-pin").forEach(pinEl => {
    const tooltip = pinEl.querySelector(".fp-dash-tooltip");

    pinEl.addEventListener("mouseenter", () => {
      const wrapRect = wrap.getBoundingClientRect();
      const dotRect = pinEl.querySelector(".fp-pin-dot").getBoundingClientRect();

      // Move tooltip to wrap for positioning in wrap coords
      tooltip.style.left = "0px";
      tooltip.style.top = "0px";
      wrap.appendChild(tooltip);
      tooltip.classList.add("fp-tt-visible");

      const ttW = tooltip.offsetWidth;
      const ttH = tooltip.offsetHeight;
      const pinCX = dotRect.left + dotRect.width / 2 - wrapRect.left;
      const pinCY = dotRect.top + dotRect.height / 2 - wrapRect.top;

      let left = pinCX - ttW / 2;
      left = Math.max(4, Math.min(left, wrapRect.width - ttW - 4));

      let top = pinCY - ttH - 16;
      if (top < 4) top = pinCY + 16;
      top = Math.max(4, Math.min(top, wrapRect.height - ttH - 4));

      tooltip.style.left = left + "px";
      tooltip.style.top = top + "px";
    });

    pinEl.addEventListener("mouseleave", () => {
      tooltip.classList.remove("fp-tt-visible");
      pinEl.appendChild(tooltip);
    });
  });

  // ── Resolve label overlaps after image loads ──
  const img = document.getElementById("fp-img-hero");
  const doLabels = () => _fpResolveLabels(wrap);
  if (img.complete) doLabels(); else img.addEventListener("load", doLabels);
}

function _renderLoadStrip(ts, totalKw, t, m) {
  const strip = document.getElementById("exec-load-strip");
  if (!strip) return;
  strip.innerHTML = `
    <div class="exec-strip-title">
      <span>Facility Load Profile</span>
      <span>Peak: ${m.peakKw.toFixed(1)} kW &nbsp; Avg: ${m.avgKw.toFixed(1)} kW</span>
    </div>
    <div id="exec-load-chart"></div>`;

  const rolling = rollingMean(ts, totalKw, D.rolling_hours);
  setTimeout(() => {
    Plotly.newPlot("exec-load-chart", [{
      x: ts, y: rolling, mode: "lines",
      line: { color: t.accent, width: 2, shape: "spline" },
      fill: "tozeroy", fillcolor: "rgba(139,212,53,0.1)",
      hovertemplate: "%{x}<br>%{y:.1f} kW<extra></extra>"
    }], pLayout({
      margin: { t: 0, l: 35, r: 8, b: 18 },
      xaxis: xA({ type: "date", showticklabels: true, tickfont: { size: 9 } }),
      yaxis: yA({ showticklabels: true, tickfont: { size: 9 } }),
      height: 55,
      hovermode: "x unified"
    }), pCfg);
  }, 0);
}

/**
 * Resolve label positions: keep next to pins, push apart if overlapping.
 */
function _fpResolveLabels(wrap) {
  const wrapW = wrap.offsetWidth;
  const wrapH = wrap.offsetHeight;
  if (!wrapW || !wrapH) return;

  const pinEls = Array.from(wrap.querySelectorAll(".fp-dash-pin"));
  if (!pinEls.length) return;

  const labels = pinEls.map(pin => {
    const tag = pin.querySelector(".fp-pin-tag");
    if (!tag) return null;
    const px = parseFloat(pin.style.left) / 100 * wrapW;
    const py = parseFloat(pin.style.top) / 100 * wrapH;
    tag.style.left = "12px";
    tag.style.top = "-3px";
    tag.style.transform = "none";
    const tw = tag.offsetWidth;
    const th = tag.offsetHeight;
    let x = px + 5;
    let y = py - 10;
    if (x + tw > wrapW - 4) { x = px - tw - 5; }
    return { el: tag, x, y, w: tw, h: th, anchorX: px, anchorY: py };
  }).filter(Boolean);

  for (let iter = 0; iter < 40; iter++) {
    let moved = false;
    for (let i = 0; i < labels.length; i++) {
      for (let j = i + 1; j < labels.length; j++) {
        const a = labels[i], b = labels[j];
        const ox = Math.min(a.x+a.w, b.x+b.w) - Math.max(a.x, b.x);
        const oy = Math.min(a.y+a.h, b.y+b.h) - Math.max(a.y, b.y);
        if (ox > 0 && oy > 0) {
          const push = (oy / 2) + 2;
          if (a.y <= b.y) { a.y -= push; b.y += push; } else { a.y += push; b.y -= push; }
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  labels.forEach(lb => {
    lb.y = Math.max(0, Math.min(lb.y, wrapH - lb.h));
    lb.x = Math.max(0, Math.min(lb.x, wrapW - lb.w));
    const offX = lb.x - (lb.anchorX - 7);
    const offY = lb.y - (lb.anchorY - 7);
    lb.el.style.left = offX + "px";
    lb.el.style.top = offY + "px";
  });
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
  Plotly.newPlot("chart-hourly",[{
    x:hp.hours,y:hp.avgs, mode:"lines+markers",
    line:{color:t.accentDark,width:2.5,shape:"spline"},
    marker:{size:7,color:t.accentDark,line:{color:t.card,width:2}},
    fill:"tozeroy",fillcolor:t.accentDark+"12"
  }],pLayout({
    xaxis:xA({title:{text:"Hour (UTC)",font:{size:12}},dtick:2}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}})
  }),pCfg);

  const wp=weekdayProfile(data.timestamps,data.totalKw);
  Plotly.newPlot("chart-weekday",[{
    x:wp.days,y:wp.avgs, type:"bar", marker:{color:t.accent}
  }],pLayout({
    xaxis:xA({title:{text:"Day",font:{size:12}}}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}}), bargap:0.2
  }),pCfg);

  const de=dailyEnergy(data.timestamps,data.totalKw);
  Plotly.newPlot("chart-daily-energy",[{
    x:de.dates,y:de.values, type:"bar", marker:{color:t.accentDark}
  }],pLayout({
    xaxis:xA({title:{text:"Date",font:{size:12}},type:"category"}),
    yaxis:yA({title:{text:"kWh",font:{size:12}}}), bargap:0.15
  }),pCfg);

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
    Plotly.newPlot("chart-panels",pShow.map(p=>({
      x:data.timestamps, y:data.panelSeries[p]||[], mode:"lines",name:p,line:{width:2,shape:"spline"}
    })),pLayout({
      legend:{orientation:"h",y:-0.15},
      xaxis:xA({title:{text:"Time",font:{size:12}},type:"date"}),
      yaxis:yA({title:{text:"kW",font:{size:12}}}),
      shapes:weekendShapes(data.timestamps)
    }),pCfg);
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

  Plotly.newPlot("chart-comp-load",[
    {x:pd1.ts,y:pd1.kw,mode:"lines",name:"Period 1",line:{color:t.accent,width:2.5,shape:"spline"}},
    {x:pd2.ts,y:pd2.kw,mode:"lines",name:"Period 2",line:{color:t.accentDark,width:2.5,shape:"spline"}}
  ],pLayout({ legend:{orientation:"h",y:-0.15},
    xaxis:xA({title:{text:"Time",font:{size:12}},type:"date"}),
    yaxis:yA({title:{text:"kW",font:{size:12}}})
  }),pCfg);

  const hp1=hourlyProfile(pd1.ts,pd1.kw),hp2=hourlyProfile(pd2.ts,pd2.kw);
  Plotly.newPlot("chart-comp-hourly",[
    {x:hp1.hours,y:hp1.avgs,mode:"lines+markers",name:"P1",line:{color:t.accent,width:2.5,shape:"spline"},marker:{size:7,color:t.accent}},
    {x:hp2.hours,y:hp2.avgs,mode:"lines+markers",name:"P2",line:{color:t.accentDark,width:2.5,shape:"spline"},marker:{size:7,color:t.accentDark}}
  ],pLayout({ legend:{orientation:"h",y:-0.15},
    xaxis:xA({title:{text:"Hour",font:{size:12}},dtick:2}),
    yaxis:yA({title:{text:"Avg kW",font:{size:12}}})
  }),pCfg);

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
  if      (tabKey === "overview")   renderOverview();
  else if (tabKey === "analytics")  renderAnalytics();
  else if (tabKey === "comparison") renderComparison();
  else if (tabKey === "data")       renderDataTable();
  else if (tabKey === "devices")    renderDevicesTab();
}

function renderCurrentTab() {
  renderTab(activeTab);
}

/* ─── Boot ─── */
document.addEventListener("DOMContentLoaded", initDashboard);
