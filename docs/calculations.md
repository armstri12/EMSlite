# EMSlite Calculation Reference

_Last reviewed: 2026-02-20_

Diagram: `docs/calculation-flow.excalidraw` visualizes the end-to-end calculation pipeline and core formulas.

This document defines every formula used in EMSlite, with units, assumptions, and worked examples. All code references are to `emslite/metrics.py`, `emslite/core.py`, `emslite/behavior.py`, and `emslite/trending.py`.

---

## 1. Amps → kW conversion

**Formula:**
```
kW = amps × (voltage × √3 × power_factor × calibration_factor) / 1000
```

**Where:**
- `amps` — measured phase current (A), one reading per sample per panel
- `voltage` — system line voltage (V), default `480` V (three-phase)
- `power_factor` — dimensionless, default `1.0`
- `calibration_factor` — dimensionless empirical trim, default `1.0`
- `√3 ≈ 1.7321`

**Code:** `core.py:amps_to_kw()`

**Worked example:**
```
amps = 100 A, voltage = 480 V, power_factor = 1.0, calibration_factor = 1.0
kW = 100 × (480 × 1.7321 × 1.0 × 1.0) / 1000
kW = 100 × 831.6 / 1000
kW = 83.16 kW
```

### Apparent vs. real power (the #1 cause of over-reporting)

Current sensors measure amperage only, so this formula computes **apparent power
(kVA)**. The utility bills **real energy (kWh)**, and `real = apparent × PF`.
Leaving `power_factor = 1.0` therefore over-reports energy by `1 / PF`:

| True facility PF | Over-report with PF=1.0 |
|------------------|-------------------------|
| 0.95             | +5 %                    |
| 0.90             | +11 %                   |
| 0.83             | +20 %                   |
| 0.80             | +25 %                   |

Set `power_factor` to the facility's measured PF (typically 0.80–0.95). Never
leave it at 1.0 in production.

### `calibration_factor`

A single empirical scalar (default `1.0`) applied inside `amps_to_kw`, so it
scales every kW/kWh/cost output uniformly. Use it to absorb any residual gap
once `power_factor`, `line_voltage`, and `aggregate_columns` (below) are correct.
`GET /api/bills/{id}/comparison` returns a `suggested_calibration_factor` derived
from a real bill, and `scripts/reconcile_bill.py` prints one for any date range.

### Per-meter overrides (`meter_overrides`)

A facility with multiple utility meters gets one bill per meter, and the meters
can reconcile to different factors (different load mix or true PF). Set
`meter_overrides` in the config, keyed by meter name (matching the device
`meter_name` field):

```jsonc
"meter_overrides": {
  "Meter A": { "calibration_factor": 1.18, "power_factor": 0.84 },
  "Meter B": { "calibration_factor": 1.05 }   // power_factor falls back to global
}
```

Each panel is resolved to its meter via `core.get_device_meter_map()`, and the
override wins over the global `power_factor` / `calibration_factor` for that
panel everywhere the facility total and KPIs are computed
(`core.build_meter_factor_maps`, threaded through `metrics.py` as `pf_map` /
`calibration_map`). `routes_bills.bill_comparison` resolves a single meter's
factors via `core.meter_factors(cfg, meter_name)`.

### On-site solar (net vs. gross energy)

When the site has solar generation, current sensors measure **gross
consumption** while the utility bills **net energy** (`net = consumption −
solar`). Comparing gross CT energy to a net bill therefore *should* show the
dashboard higher than the bill, by roughly the solar generated — this is
expected, not an error.

Record solar per bill via the `solar_kwh` field on a `UtilityBill`
(`POST`/`PUT /api/bills`). `bill_comparison` then reports `total_kwh` (gross),
`solar_kwh`, and `net_kwh = total_kwh − solar_kwh`, compares `net_kwh` to the
bill, and derives `suggested_calibration_factor` against the target **gross**
consumption (`billed_energy + solar_kwh`). The reconciliation script takes
`--solar-kwh`.

For an apples-to-apples energy check, record the **net kWh printed on the bill**
via the `billed_kwh` field on a `UtilityBill`. When present, `bill_comparison`
uses it as `billed_energy` directly (returning `billed_kwh` and
`kwh_difference = net_kwh − billed_kwh`) instead of inferring it from
`amount / price_per_kwh`, which makes both the difference and the
`suggested_calibration_factor` independent of the price assumption. The
reconciliation script takes the equivalent `--bill-kwh`.

### Avoiding double-counting (`aggregate_columns`)

The facility total sums every meter column except those returned by
`core.excluded_columns(cfg)` — namely `combo_columns` keys and any
`aggregate_columns`. If the master CSV contains a **main-feed / sub-total CT
column** (e.g. `Main_Service`) that already includes its branch panels, summing
it on top of the branches double-counts. List such columns in `aggregate_columns`
to exclude them. `scripts/reconcile_bill.py` flags suspected aggregate columns.

**Assumptions and caveats:**
- Assumes balanced three-phase load. Single-phase or unbalanced loads will be misrepresented.
- Applied identically in `metrics.py`, `behavior.py`, `trending.py`, `report.py`, and the `routes_*` API modules.
- Config edits take effect via `PUT /api/config` (the dashboard's Settings panel); the in-memory config is also refreshed there. Hand-editing `visualization_config.json` while the server runs requires a restart.

---

## 2. Energy integration (kWh)

**Formula:**
```
kWh = Σ ((kW_i + kW_{i-1}) / 2) × dt_i
```

where `dt_i` is the time elapsed from the previous sample to sample `i` (in hours).

**Integration method:** Trapezoidal rule — each interval contributes the average
of its two endpoint kW values times the interval length:
`kWh = Σ ((kW_i + kW_{i-1}) / 2) × dt_i`. This removes the systematic bias of
right-endpoint summation (which weighted the later reading across the whole
interval). The difference versus the old method is typically <1–2 %.

**Code:** `metrics.py:integrate_kwh()` (used by `compute_kpi`, `compute_panel_rankings`, `compute_department_breakdown`), and `routes_bills.py:bill_comparison()`.

**Worked example (uniform cadence):**
```
Samples at 15-minute intervals (dt = 0.25 h):
  t=0:00  kW=50   dt=0.00 h  → 0.00 kWh
  t=0:15  kW=60   dt=0.25 h  → ((50+60)/2) × 0.25 = 13.75 kWh
  t=0:30  kW=40   dt=0.25 h  → ((60+40)/2) × 0.25 = 12.50 kWh

kWh = 0.00 + 13.75 + 12.50 = 26.25 kWh
```

Note: the first sample has `dt = 0` (no previous sample), so it contributes 0 kWh.

**Worked example (irregular cadence):**
```
  t=0:00  kW=50   dt=0.00 h  → 0.00 kWh
  t=0:15  kW=60   dt=0.25 h  → ((50+60)/2) × 0.25 = 13.75 kWh
  t=1:00  kW=40   dt=0.75 h  → ((60+40)/2) × 0.75 = 37.50 kWh
  t=1:15  kW=55   dt=0.25 h  → ((40+55)/2) × 0.25 = 11.875 kWh

Total = 63.125 kWh
```

**Relationship to average kW:** With irregular cadence, the simple arithmetic mean of kW values is NOT the same as `total_kwh / total_hours`. See §3.

---

## 3. Average kW (time-weighted)

**Formula:**
```
avg_kW = Σ(kW_i × dt_i) / Σ(dt_i)
       = total_kWh / total_hours
```

**Code:** `metrics.py:compute_kpi()` — uses time-weighted formula

**Why not arithmetic mean?**
The simple mean `kW.mean()` assigns equal weight to every sample regardless of how long that power level was sustained. With irregular cadence (e.g., a 45-minute gap followed by a 15-minute gap), the simple mean overweights readings from short intervals.

**Worked example showing divergence:**
```
Samples:
  t=0:00  kW=10   dt=0.00 h
  t=0:15  kW=10   dt=0.25 h  → 2.5 kWh
  t=1:00  kW=90   dt=0.75 h  → 67.5 kWh

Simple mean = (10 + 10 + 90) / 3 = 36.7 kW
Time-weighted mean = (2.5 + 67.5) / (0.25 + 0.75) = 70.0 / 1.0 = 70.0 kW

The simple mean dramatically underrepresents the dominant 90 kW load.
```

**Load factor:**
```
load_factor = avg_kW / peak_kW × 100%
```

A high load factor (close to 100%) means load is steady. A low load factor means there are significant spikes relative to average consumption.

---

## 4. Cost and carbon

**Formulas:**
```
total_cost ($) = total_kWh × price_per_kWh
total_carbon_kg = total_kWh × carbon_kg_per_kWh
total_carbon_tonnes = total_carbon_kg / 1000
```

**Defaults:** `price_per_kwh = 0.25` $/kWh, `carbon_kg_per_kwh = 0.4` kg/kWh

**Code:** `metrics.py:compute_kpi()`

---

## 5. Period-over-period percent change (trending)

**Formula:**
```
pct_change = (recent_kWh - prior_kWh) / prior_kWh × 100
```

**Code:** `trending.py:compute_trending_snapshot()` lines 83-88 (per-panel), lines 135-140 (facility)

**Edge case policy (unified for panel and facility):**

| prior_kWh | recent_kWh | result |
|-----------|------------|--------|
| > 0 | any | `(recent - prior) / prior × 100` |
| 0 | > 0 | `100.0` (new energy appeared; treated as full increase) |
| 0 | 0 | `0.0` (no change) |

**Trend direction thresholds:**
- `pct_change > +5%` → `"rising"`
- `pct_change < -5%` → `"falling"`
- `-5% ≤ pct_change ≤ +5%` → `"stable"`

---

## 6. Shift vs off-shift energy split

**Shift definition:**
- Weekdays (Mon–Fri), 6:00 AM – 2:30 PM America/New_York
- All other times are "off-shift"

**Code:** `behavior.py:classify_shift_hours()`, `compute_shift_energy_split()`

**Formula:**
```
shift_kWh = Σ(kW_i × dt_i) for samples where shift_class == "shift"
off_shift_kWh = total_kWh - shift_kWh
shift_pct = shift_kWh / total_kWh × 100
```

---

## 7. Phantom draw

**Definition:** Baseline power drawn when the facility is empty (deep off-hours).

**Deep off-hours:** Weeknights, 11 PM–4 AM Eastern.

**Formula:**
```
phantom_kW = 25th percentile of kW during deep off-hours
```

The 25th percentile is used instead of the minimum to be robust against sensor noise and zero-readings from momentary outages.

**Code:** `behavior.py:compute_phantom_draw()`

**Annualized phantom cost:**
```
off_shift_hours_per_year = (15.5 h/workday × 260 workdays) + (24 h/day × 105 weekend days)
                         = 4,030 + 2,520 = 6,550 hours

annual_phantom_kWh = phantom_kW × 6,550
annual_phantom_cost = annual_phantom_kWh × price_per_kWh
```

**Code:** `behavior.py:compute_annualized_phantom_cost()`

---

## 8. Total kW and combo columns

The `total_kw` field in `/api/data` responses is the sum of physical meter columns only. Configured `combo_columns` (pre-aggregated groups) are excluded from this sum and returned separately in `group_series` to avoid double-counting.

**Code:** `routes_data.py` — `meter_columns(exclude=combo_columns)` on line 60, separate `group_series` computation on lines 91-97.

**Implication for users:** The sum of `group_series` values will NOT equal `total_kw` unless the groups exactly partition all panels with no overlap. This is by design. `total_kw` reflects the physical panel-level sum; `group_series` reflects pre-computed aggregates from the source data.
