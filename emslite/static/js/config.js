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
