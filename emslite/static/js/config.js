/**
 * EMSlite Device & Department Configuration Screen
 */

let devicesData = [];
let departmentsData = [];
let selectedDevices = new Set();

async function renderDevicesTab() {
  try {
    [devicesData, departmentsData] = await Promise.all([
      API.getDevices(),
      API.getDepartments(),
    ]);
    renderDeviceTable();
    renderDepartmentSection();
    renderFloorPlanList();
  } catch (err) {
    console.error("Failed to load device config:", err);
  }
}

/* ═══════════════════════════════════════════════════════
   DEVICE TABLE
   ═══════════════════════════════════════════════════════ */
function renderDeviceTable() {
  const container = document.getElementById("device-table-container");
  if (!container) return;

  const deptFilter = document.getElementById("device-dept-filter");
  const filterVal = deptFilter ? deptFilter.value : "";

  let filtered = devicesData;
  if (filterVal) {
    filtered = devicesData.filter(d => d.department_id === filterVal);
  }

  let html = `
    <div class="table-controls" style="margin-bottom:12px">
      <select id="device-dept-filter" class="filter-input" style="min-width:160px">
        <option value="">All Departments</option>
        ${departmentsData.map(d => `<option value="${d.id}" ${d.id === filterVal ? 'selected' : ''}>${d.display_name}</option>`).join("")}
      </select>
      <button class="btn btn-ghost btn-sm" id="bulk-assign-btn" style="margin-left:auto" disabled>Bulk Assign</button>
    </div>
    <div class="data-table-wrap">
      <table class="data-table" id="device-list-table">
        <thead><tr>
          <th style="width:40px"><input type="checkbox" id="select-all-devices"/></th>
          <th>Display Name</th>
          <th>ID (CSV Column)</th>
          <th>Department</th>
          <th>Location</th>
          <th>Type</th>
          <th>Enabled</th>
          <th>Actions</th>
        </tr></thead>
        <tbody>
          ${filtered.map(d => `<tr>
            <td><input type="checkbox" class="device-cb" value="${d.id}" ${selectedDevices.has(d.id) ? 'checked' : ''}/></td>
            <td style="font-weight:600;color:var(--heading)">${d.display_name}</td>
            <td style="font-size:0.75rem;color:var(--muted)">${d.id}</td>
            <td>${d.department_name || '<span style="color:var(--muted)">Unassigned</span>'}</td>
            <td>${d.location || '<span style="color:var(--muted)">-</span>'}</td>
            <td>${d.device_type}</td>
            <td>${d.enabled ? '<span style="color:var(--positive-text)">Yes</span>' : '<span style="color:var(--negative-text)">No</span>'}</td>
            <td><button class="btn btn-ghost btn-sm edit-device-btn" data-id="${d.id}">Edit</button></td>
          </tr>`).join("")}
        </tbody>
      </table>
      <div class="table-footer">
        <span>${filtered.length} device${filtered.length !== 1 ? 's' : ''}</span>
      </div>
    </div>`;

  container.innerHTML = html;

  // Wire events
  document.getElementById("device-dept-filter").addEventListener("change", () => renderDeviceTable());
  document.getElementById("select-all-devices").addEventListener("change", (e) => {
    selectedDevices.clear();
    if (e.target.checked) filtered.forEach(d => selectedDevices.add(d.id));
    renderDeviceTable();
  });
  document.querySelectorAll(".device-cb").forEach(cb => {
    cb.addEventListener("change", () => {
      cb.checked ? selectedDevices.add(cb.value) : selectedDevices.delete(cb.value);
      document.getElementById("bulk-assign-btn").disabled = selectedDevices.size === 0;
    });
  });
  document.getElementById("bulk-assign-btn").addEventListener("click", showBulkAssignModal);
  document.querySelectorAll(".edit-device-btn").forEach(btn => {
    btn.addEventListener("click", () => showDeviceEditPanel(btn.dataset.id));
  });
  document.getElementById("bulk-assign-btn").disabled = selectedDevices.size === 0;
}

/* ═══════════════════════════════════════════════════════
   DEVICE EDIT PANEL
   ═══════════════════════════════════════════════════════ */
async function showDeviceEditPanel(deviceId) {
  const device = devicesData.find(d => d.id === deviceId);
  if (!device) return;

  const overlay = document.createElement("div");
  overlay.className = "sidebar-overlay active";
  overlay.style.zIndex = "400";

  const panel = document.createElement("div");
  panel.style.cssText = "position:fixed;right:0;top:0;width:420px;height:100vh;background:var(--card-bg);z-index:401;overflow-y:auto;box-shadow:-4px 0 24px rgba(0,0,0,0.2);padding:2rem;";
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem">
      <h3 style="color:var(--heading);font-size:1.125rem">Edit Device</h3>
      <button class="btn btn-ghost btn-sm" id="close-edit">Close</button>
    </div>
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Display Name</label>
        <input type="text" class="filter-input" id="edit-display-name" value="${device.display_name}" style="width:100%;margin-top:4px"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Department</label>
        <select class="filter-input" id="edit-department" style="width:100%;margin-top:4px">
          <option value="">Unassigned</option>
          ${departmentsData.map(d => `<option value="${d.id}" ${d.id === device.department_id ? 'selected' : ''}>${d.display_name}</option>`).join("")}
        </select>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Location</label>
        <input type="text" class="filter-input" id="edit-location" value="${device.location || ''}" style="width:100%;margin-top:4px" placeholder="Building A, Room 2"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Device Type</label>
        <select class="filter-input" id="edit-device-type" style="width:100%;margin-top:4px">
          ${["panel","meter","transformer","mcc","vfd","other"].map(t => `<option value="${t}" ${t === device.device_type ? 'selected' : ''}>${t}</option>`).join("")}
        </select>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Rated Capacity (kW)</label>
        <input type="number" class="filter-input" id="edit-rated-capacity" value="${device.rated_capacity || ''}" style="width:100%;margin-top:4px"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Voltage Override</label>
        <input type="number" class="filter-input" id="edit-voltage" value="${device.voltage || ''}" style="width:100%;margin-top:4px" placeholder="Leave blank for global default"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Phase</label>
        <select class="filter-input" id="edit-phase" style="width:100%;margin-top:4px">
          <option value="3-phase" ${device.phase === '3-phase' ? 'selected' : ''}>3-phase</option>
          <option value="single-phase" ${device.phase === 'single-phase' ? 'selected' : ''}>Single-phase</option>
        </select>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Tags (comma-separated)</label>
        <input type="text" class="filter-input" id="edit-tags" value="${(device.tags || []).join(', ')}" style="width:100%;margin-top:4px" placeholder="critical, hvac, line-1"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Notes</label>
        <textarea class="filter-input" id="edit-notes" style="width:100%;margin-top:4px;min-height:80px;resize:vertical">${device.notes || ''}</textarea>
      </div>
      <div style="display:flex;gap:1rem">
        <div style="flex:1">
          <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Warning kW</label>
          <input type="number" class="filter-input" id="edit-warning-kw" value="${device.warning_kw || ''}" style="width:100%;margin-top:4px"/>
        </div>
        <div style="flex:1">
          <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Critical kW</label>
          <input type="number" class="filter-input" id="edit-critical-kw" value="${device.critical_kw || ''}" style="width:100%;margin-top:4px"/>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="edit-enabled" ${device.enabled ? 'checked' : ''} style="accent-color:var(--p-400)"/>
        <label style="font-size:0.875rem;color:var(--heading)">Enabled (include in calculations)</label>
      </div>
      <button class="btn btn-primary" id="save-device" style="width:100%;margin-top:0.5rem">Save Changes</button>
    </div>`;

  document.body.appendChild(overlay);
  document.body.appendChild(panel);

  function closePanel() { overlay.remove(); panel.remove(); }
  overlay.addEventListener("click", closePanel);
  document.getElementById("close-edit").addEventListener("click", closePanel);

  document.getElementById("save-device").addEventListener("click", async () => {
    const tagsRaw = document.getElementById("edit-tags").value;
    const tags = tagsRaw ? tagsRaw.split(",").map(t => t.trim()).filter(Boolean) : [];
    const data = {
      display_name: document.getElementById("edit-display-name").value,
      department_id: document.getElementById("edit-department").value || null,
      location: document.getElementById("edit-location").value || null,
      device_type: document.getElementById("edit-device-type").value,
      rated_capacity: parseFloat(document.getElementById("edit-rated-capacity").value) || null,
      voltage: parseFloat(document.getElementById("edit-voltage").value) || null,
      phase: document.getElementById("edit-phase").value,
      tags: tags,
      notes: document.getElementById("edit-notes").value || null,
      warning_kw: parseFloat(document.getElementById("edit-warning-kw").value) || null,
      critical_kw: parseFloat(document.getElementById("edit-critical-kw").value) || null,
      enabled: document.getElementById("edit-enabled").checked,
    };
    await API.updateDevice(deviceId, data);
    closePanel();
    renderDevicesTab();
  });
}

/* ═══════════════════════════════════════════════════════
   BULK ASSIGN MODAL
   ═══════════════════════════════════════════════════════ */
function showBulkAssignModal() {
  const overlay = document.createElement("div");
  overlay.className = "sidebar-overlay active";
  overlay.style.zIndex = "400";

  const modal = document.createElement("div");
  modal.style.cssText = "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--card-bg);z-index:401;border-radius:var(--card-radius);padding:2rem;min-width:360px;box-shadow:0 12px 48px rgba(0,0,0,0.2);";
  modal.innerHTML = `
    <h3 style="color:var(--heading);margin-bottom:1rem">Bulk Assign (${selectedDevices.size} devices)</h3>
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Department</label>
        <select class="filter-input" id="bulk-department" style="width:100%;margin-top:4px">
          <option value="">-- No change --</option>
          <option value="__none__">Unassign</option>
          ${departmentsData.map(d => `<option value="${d.id}">${d.display_name}</option>`).join("")}
        </select>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Location</label>
        <input type="text" class="filter-input" id="bulk-location" style="width:100%;margin-top:4px" placeholder="Leave blank for no change"/>
      </div>
      <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:0.5rem">
        <button class="btn btn-ghost" id="bulk-cancel">Cancel</button>
        <button class="btn btn-primary" id="bulk-apply">Apply</button>
      </div>
    </div>`;

  document.body.appendChild(overlay);
  document.body.appendChild(modal);

  function closeModal() { overlay.remove(); modal.remove(); }
  overlay.addEventListener("click", closeModal);
  document.getElementById("bulk-cancel").addEventListener("click", closeModal);

  document.getElementById("bulk-apply").addEventListener("click", async () => {
    const deptVal = document.getElementById("bulk-department").value;
    const locVal = document.getElementById("bulk-location").value;
    const payload = { device_ids: Array.from(selectedDevices) };
    if (deptVal === "__none__") payload.department_id = "";
    else if (deptVal) payload.department_id = deptVal;
    if (locVal) payload.location = locVal;
    await API.bulkAssignDevices(payload);
    selectedDevices.clear();
    closeModal();
    renderDevicesTab();
  });
}

/* ═══════════════════════════════════════════════════════
   DEPARTMENT MANAGEMENT
   ═══════════════════════════════════════════════════════ */
function renderDepartmentSection() {
  const container = document.getElementById("department-section");
  if (!container) return;

  container.innerHTML = `
    <div class="panel" style="margin-bottom:var(--card-gap)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
        <div class="panel-title" style="margin-bottom:0">Departments</div>
        <button class="btn btn-primary btn-sm" id="add-dept-btn">+ Add Department</button>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:12px" id="dept-cards">
        ${departmentsData.map(d => `
          <div style="background:var(--bar-track);border-radius:0.75rem;padding:1rem 1.25rem;min-width:180px;display:flex;align-items:center;gap:12px;border:2px solid var(--card-border)">
            <div style="width:12px;height:12px;border-radius:50%;background:${d.color};flex-shrink:0"></div>
            <div style="flex:1">
              <div style="font-weight:600;color:var(--heading)">${d.display_name}</div>
              <div style="font-size:0.75rem;color:var(--muted)">${d.device_count} device${d.device_count !== 1 ? 's' : ''}</div>
            </div>
            <button class="btn btn-ghost btn-sm edit-dept-btn" data-id="${d.id}" style="padding:4px 8px">Edit</button>
            <button class="btn btn-ghost btn-sm del-dept-btn" data-id="${d.id}" style="padding:4px 8px;color:var(--negative-text)">Del</button>
          </div>`).join("")}
      </div>
    </div>`;

  document.getElementById("add-dept-btn").addEventListener("click", showAddDepartmentModal);
  document.querySelectorAll(".edit-dept-btn").forEach(btn => {
    btn.addEventListener("click", () => showEditDepartmentModal(btn.dataset.id));
  });
  document.querySelectorAll(".del-dept-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (confirm("Delete this department? Devices will be unassigned.")) {
        await API.deleteDepartment(btn.dataset.id);
        renderDevicesTab();
      }
    });
  });
}

function showAddDepartmentModal() {
  showDeptModal(null);
}

function showEditDepartmentModal(deptId) {
  const dept = departmentsData.find(d => d.id === deptId);
  showDeptModal(dept);
}

function showDeptModal(dept) {
  const isEdit = !!dept;
  const overlay = document.createElement("div");
  overlay.className = "sidebar-overlay active";
  overlay.style.zIndex = "400";

  const modal = document.createElement("div");
  modal.style.cssText = "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--card-bg);z-index:401;border-radius:var(--card-radius);padding:2rem;min-width:360px;box-shadow:0 12px 48px rgba(0,0,0,0.2);";
  modal.innerHTML = `
    <h3 style="color:var(--heading);margin-bottom:1rem">${isEdit ? 'Edit' : 'Add'} Department</h3>
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Name</label>
        <input type="text" class="filter-input" id="dept-name" value="${dept ? dept.display_name : ''}" style="width:100%;margin-top:4px"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Color</label>
        <input type="color" id="dept-color" value="${dept ? dept.color : '#8BD435'}" style="margin-top:4px;width:60px;height:36px;border:none;cursor:pointer"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Description</label>
        <input type="text" class="filter-input" id="dept-desc" value="${dept ? (dept.description || '') : ''}" style="width:100%;margin-top:4px"/>
      </div>
      <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:0.5rem">
        <button class="btn btn-ghost" id="dept-cancel">Cancel</button>
        <button class="btn btn-primary" id="dept-save">${isEdit ? 'Update' : 'Create'}</button>
      </div>
    </div>`;

  document.body.appendChild(overlay);
  document.body.appendChild(modal);

  function closeModal() { overlay.remove(); modal.remove(); }
  overlay.addEventListener("click", closeModal);
  document.getElementById("dept-cancel").addEventListener("click", closeModal);

  document.getElementById("dept-save").addEventListener("click", async () => {
    const data = {
      display_name: document.getElementById("dept-name").value,
      color: document.getElementById("dept-color").value,
      description: document.getElementById("dept-desc").value || null,
    };
    if (isEdit) {
      await API.updateDepartment(dept.id, data);
    } else {
      await API.createDepartment(data);
    }
    closeModal();
    renderDevicesTab();
  });
}

/* ═══════════════════════════════════════════════════════
   FLOOR PLAN MANAGEMENT
   ═══════════════════════════════════════════════════════ */

async function renderFloorPlanList() {
  const container = document.getElementById("floorplan-list");
  if (!container) return;

  const plans = await API.getFloorPlans();
  if (!plans.length) {
    container.innerHTML = '<div style="color:var(--muted);padding:1rem;text-align:center">No floor plans uploaded yet.</div>';
  } else {
    container.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:16px;padding:1rem 0">
      ${plans.map(fp => `
        <div class="fp-card" style="background:var(--bar-track);border-radius:0.75rem;padding:1rem;min-width:220px;border:2px solid var(--card-border);cursor:pointer" data-id="${fp.id}">
          <div style="font-weight:600;color:var(--heading)">${fp.name}</div>
          <div style="font-size:0.75rem;color:var(--muted);margin-top:4px">${fp.pin_count} device${fp.pin_count !== 1 ? 's' : ''} placed</div>
          <div style="font-size:0.75rem;margin-top:4px;color:${fp.show_on_dashboard ? 'var(--positive-text)' : 'var(--muted)'}">
            ${fp.show_on_dashboard ? 'Shown on Dashboard' : 'Not on Dashboard'}
          </div>
          <div style="display:flex;gap:6px;margin-top:8px">
            <button class="btn btn-primary btn-sm fp-edit-btn" data-id="${fp.id}">Edit Pins</button>
            <button class="btn btn-ghost btn-sm fp-toggle-btn" data-id="${fp.id}" data-show="${fp.show_on_dashboard}">${fp.show_on_dashboard ? 'Hide from Dashboard' : 'Show on Dashboard'}</button>
            <button class="btn btn-ghost btn-sm fp-del-btn" data-id="${fp.id}" style="color:var(--negative-text)">Delete</button>
          </div>
        </div>
      `).join("")}
    </div>`;
  }

  // Upload button
  document.getElementById("btn-add-floorplan").addEventListener("click", showFloorPlanUploadModal);

  document.querySelectorAll(".fp-edit-btn").forEach(btn => {
    btn.addEventListener("click", e => { e.stopPropagation(); openFloorPlanEditor(parseInt(btn.dataset.id)); });
  });
  document.querySelectorAll(".fp-toggle-btn").forEach(btn => {
    btn.addEventListener("click", async e => {
      e.stopPropagation();
      await API.updateFloorPlan(parseInt(btn.dataset.id), { show_on_dashboard: btn.dataset.show !== "true" });
      renderFloorPlanList();
    });
  });
  document.querySelectorAll(".fp-del-btn").forEach(btn => {
    btn.addEventListener("click", async e => {
      e.stopPropagation();
      if (confirm("Delete this floor plan and all its pins?")) {
        await API.deleteFloorPlan(parseInt(btn.dataset.id));
        renderFloorPlanList();
      }
    });
  });
}

function showFloorPlanUploadModal() {
  const overlay = document.createElement("div");
  overlay.className = "sidebar-overlay active";
  overlay.style.zIndex = "400";

  const modal = document.createElement("div");
  modal.style.cssText = "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--card-bg);z-index:401;border-radius:var(--card-radius);padding:2rem;min-width:400px;box-shadow:0 12px 48px rgba(0,0,0,0.2);";
  modal.innerHTML = `
    <h3 style="color:var(--heading);margin-bottom:1rem">Upload Floor Plan</h3>
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Name</label>
        <input type="text" class="filter-input" id="fp-name" placeholder="e.g. Main Building - Level 1" style="width:100%;margin-top:4px"/>
      </div>
      <div>
        <label style="font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase">Image</label>
        <input type="file" id="fp-image" accept="image/*" style="margin-top:4px;color:var(--heading)"/>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="fp-show-dash" style="accent-color:var(--p-400)"/>
        <label style="font-size:0.875rem;color:var(--heading)">Show on Dashboard</label>
      </div>
      <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:0.5rem">
        <button class="btn btn-ghost" id="fp-cancel">Cancel</button>
        <button class="btn btn-primary" id="fp-upload">Upload</button>
      </div>
    </div>`;

  document.body.appendChild(overlay);
  document.body.appendChild(modal);

  function closeModal() { overlay.remove(); modal.remove(); }
  overlay.addEventListener("click", closeModal);
  document.getElementById("fp-cancel").addEventListener("click", closeModal);

  document.getElementById("fp-upload").addEventListener("click", async () => {
    const name = document.getElementById("fp-name").value.trim();
    const file = document.getElementById("fp-image").files[0];
    if (!name || !file) { alert("Please provide a name and image."); return; }
    const fd = new FormData();
    fd.append("name", name);
    fd.append("image", file);
    fd.append("show_on_dashboard", document.getElementById("fp-show-dash").checked);
    await API.createFloorPlan(fd);
    closeModal();
    renderFloorPlanList();
  });
}

/* ═══════════════════════════════════════════════════════
   FLOOR PLAN EDITOR (place/move/remove device pins)
   ═══════════════════════════════════════════════════════ */

async function openFloorPlanEditor(planId) {
  const plan = await API.getFloorPlan(planId);
  const editor = document.getElementById("floorplan-editor");
  editor.classList.remove("hidden");

  const allDevices = devicesData || await API.getDevices();
  let zones = plan.zones || [];
  let drawingDevice = null;  // device_id being drawn
  let drawingPts = [];       // in-progress polygon vertices [{x, y}]

  const zoneColors = ["#8BD435","#398EB9","#F97316","#EF4444","#38BDF8","#6DBF1A","#1D5B83","#9333EA"];

  // ── Update just the SVG overlay (no full DOM rebuild) ──
  function updateSvg() {
    const svg = document.getElementById("fp-svg");
    if (!svg) return;
    svg.innerHTML = "";
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");

    // Draw saved zones
    zones.forEach((z, idx) => {
      const pts = z.points || [];
      if (pts.length < 3) return;
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("points", pts.map(p => `${p.x},${p.y}`).join(" "));
      poly.setAttribute("fill", zoneColors[idx % zoneColors.length]);
      poly.setAttribute("fill-opacity", "0.25");
      poly.setAttribute("stroke", zoneColors[idx % zoneColors.length]);
      poly.setAttribute("stroke-width", "0.4");
      poly.setAttribute("stroke-opacity", "0.8");
      svg.appendChild(poly);
      const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
      const dev = allDevices.find(d => d.id === z.device_id);
      const label = z.label || (dev ? dev.display_name : z.device_id);
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", cx); text.setAttribute("y", cy);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("dominant-baseline", "central");
      text.setAttribute("font-size", "2.5"); text.setAttribute("font-weight", "700");
      text.setAttribute("fill", zoneColors[idx % zoneColors.length]);
      text.textContent = label;
      svg.appendChild(text);
    });

    // Draw in-progress polygon
    if (drawingPts.length > 0) {
      if (drawingPts.length >= 2) {
        const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyline.setAttribute("points", drawingPts.map(p => `${p.x},${p.y}`).join(" "));
        polyline.setAttribute("fill", "none");
        polyline.setAttribute("stroke", "#F97316");
        polyline.setAttribute("stroke-width", "0.4");
        polyline.setAttribute("stroke-dasharray", "1,0.5");
        svg.appendChild(polyline);
      }
      drawingPts.forEach(p => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", p.x); circle.setAttribute("cy", p.y);
        circle.setAttribute("r", "0.7"); circle.setAttribute("fill", "#F97316");
        svg.appendChild(circle);
      });
    }
  }

  // ── Update the drawing toolbar (hint, buttons) without full rebuild ──
  function updateToolbar() {
    const hint = document.getElementById("fp-draw-hint");
    const finishBtn = document.getElementById("fp-finish-draw");
    const cancelBtn = document.getElementById("fp-cancel-draw");
    const ptCount = document.getElementById("fp-pt-count");
    if (hint) hint.style.display = drawingDevice ? "" : "none";
    if (finishBtn) finishBtn.style.display = (drawingDevice && drawingPts.length >= 3) ? "" : "none";
    if (cancelBtn) cancelBtn.style.display = drawingDevice ? "" : "none";
    if (ptCount) ptCount.textContent = drawingPts.length ? `(${drawingPts.length} pts)` : "";
  }

  function render() {
    const placedIds = new Set(zones.map(z => z.device_id));
    const available = allDevices.filter(d => !placedIds.has(d.id));

    editor.innerHTML = `
      <div style="border:2px solid var(--card-border);border-radius:0.75rem;padding:1rem;margin-top:1rem">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
          <h3 style="color:var(--heading);font-size:1rem;margin:0">${plan.name} — Zone Editor</h3>
          <button class="btn btn-ghost btn-sm" id="fp-editor-close">Close Editor</button>
        </div>
        <div style="display:flex;gap:0.75rem;margin-bottom:0.75rem;align-items:center;flex-wrap:wrap">
          <select class="filter-input" id="fp-device-select" style="min-width:200px">
            <option value="">— Select device to map —</option>
            ${available.map(d => `<option value="${d.id}"${d.id === drawingDevice ? " selected" : ""}>${d.display_name} (${d.id})</option>`).join("")}
          </select>
          <span id="fp-draw-hint" style="font-size:0.78rem;color:var(--muted);display:${drawingDevice ? 'inline' : 'none'}">
            Click to add vertices. <span id="fp-pt-count">${drawingPts.length ? `(${drawingPts.length} pts)` : ""}</span>
          </span>
          <button class="btn btn-primary btn-sm" id="fp-finish-draw" style="display:${drawingDevice && drawingPts.length >= 3 ? 'inline-block' : 'none'}">Finish Zone</button>
          <button class="btn btn-ghost btn-sm" id="fp-cancel-draw" style="display:${drawingDevice ? 'inline-block' : 'none'};color:var(--negative-text)">Cancel</button>
        </div>
        <div id="fp-canvas-wrap" style="position:relative;display:inline-block;max-width:100%;border:1px solid var(--card-border);border-radius:8px;overflow:hidden;cursor:${drawingDevice ? 'crosshair' : 'default'}">
          <img src="${plan.image_path}" style="display:block;max-width:100%;height:auto" id="fp-img"/>
          <svg id="fp-svg" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none"></svg>
        </div>
        <div style="margin-top:1rem">
          <div style="font-size:0.8rem;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:6px">Defined Zones (${zones.length})</div>
          ${zones.length ? zones.map(z => {
            const dev = allDevices.find(d => d.id === z.device_id);
            const pts = z.points || [];
            return `<div style="display:flex;align-items:center;gap:8px;padding:4px 0">
              <span style="width:10px;height:10px;border-radius:3px;background:var(--p-400);flex-shrink:0"></span>
              <span style="font-size:0.85rem;color:var(--heading);flex:1">${z.label || (dev ? dev.display_name : z.device_id)}</span>
              <span style="font-size:0.7rem;color:var(--muted)">${pts.length} pts</span>
              <button class="btn btn-ghost btn-sm fp-rm-zone" data-zone-id="${z.id}" style="padding:2px 6px;color:var(--negative-text);font-size:0.75rem">Remove</button>
            </div>`;
          }).join("") : '<div style="color:var(--muted);font-size:0.85rem">No zones defined yet. Select a device and draw its coverage area.</div>'}
        </div>
      </div>`;

    // Draw SVG
    updateSvg();

    // Wire close
    document.getElementById("fp-editor-close").addEventListener("click", () => {
      editor.classList.add("hidden");
      editor.innerHTML = "";
    });

    // Wire cancel draw
    document.getElementById("fp-cancel-draw").addEventListener("click", () => {
      drawingDevice = null;
      drawingPts = [];
      render();
    });

    // Wire finish draw
    document.getElementById("fp-finish-draw").addEventListener("click", async () => {
      if (!drawingDevice || drawingPts.length < 3) return;
      const newZone = await API.addFloorPlanZone(planId, {
        device_id: drawingDevice,
        points: drawingPts,
      });
      zones.push(newZone);
      drawingDevice = null;
      drawingPts = [];
      render();
    });

    // Wire device select
    const sel = document.getElementById("fp-device-select");
    sel.addEventListener("change", () => {
      drawingDevice = sel.value || null;
      drawingPts = [];
      const wrap = document.getElementById("fp-canvas-wrap");
      wrap.style.cursor = drawingDevice ? "crosshair" : "default";
      updateToolbar();
      updateSvg();
    });

    // Wire click on canvas to add polygon vertex (NO full render — just update SVG)
    // Snap to nearest 45° angle relative to previous point
    function snapTo45(prev, rawX, rawY) {
      if (!prev) return { x: rawX, y: rawY };
      const dx = rawX - prev.x;
      const dy = rawY - prev.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 0.5) return { x: rawX, y: rawY };
      const angle = Math.atan2(dy, dx);
      const snap = Math.round(angle / (Math.PI / 4)) * (Math.PI / 4);
      return {
        x: Math.round((prev.x + dist * Math.cos(snap)) * 10) / 10,
        y: Math.round((prev.y + dist * Math.sin(snap)) * 10) / 10,
      };
    }
    const wrap = document.getElementById("fp-canvas-wrap");
    wrap.addEventListener("click", (e) => {
      if (!drawingDevice) return;
      const img = document.getElementById("fp-img");
      const rect = img.getBoundingClientRect();
      const rawX = Math.round((e.clientX - rect.left) / rect.width * 1000) / 10;
      const rawY = Math.round((e.clientY - rect.top) / rect.height * 1000) / 10;
      if (rawX < 0 || rawX > 100 || rawY < 0 || rawY > 100) return;
      const prev = drawingPts.length ? drawingPts[drawingPts.length - 1] : null;
      const { x, y } = snapTo45(prev, rawX, rawY);
      drawingPts.push({ x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) });
      updateSvg();
      updateToolbar();
    });

    // Wire remove zone buttons
    document.querySelectorAll(".fp-rm-zone").forEach(btn => {
      btn.addEventListener("click", async () => {
        const zoneId = parseInt(btn.dataset.zoneId);
        await API.deleteFloorPlanZone(planId, zoneId);
        zones = zones.filter(z => z.id !== zoneId);
        render();
      });
    });
  }

  render();
}
