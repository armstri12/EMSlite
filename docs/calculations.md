# EMSlite Calculation Reference

_Last reviewed: 2026-02-20_

This document defines every formula used in EMSlite, with units, assumptions, and worked examples. All code references are to `emslite/metrics.py`, `emslite/core.py`, `emslite/behavior.py`, and `emslite/trending.py`.

---

## 1. Amps → kW conversion

**Formula:**
```
kW = amps × (voltage × √3 × power_factor) / 1000
```

**Where:**
- `amps` — measured phase current (A), one reading per sample per panel
- `voltage` — system line voltage (V), default `480` V (three-phase)
- `power_factor` — dimensionless, default `1.0`
- `√3 ≈ 1.7321`

**Code:** `core.py:amps_to_kw()`

**Worked example:**
```
amps = 100 A, voltage = 480 V, power_factor = 1.0
kW = 100 × (480 × 1.7321 × 1.0) / 1000
kW = 100 × 831.6 / 1000
kW = 83.16 kW
```

**Assumptions and caveats:**
- Assumes balanced three-phase load. Single-phase or unbalanced loads will be misrepresented.
- `power_factor = 1.0` (unity) is the default. If the facility has significant reactive loads, update `power_factor` in `visualization_config.json`.
- Applied identically in `metrics.py`, `behavior.py`, `trending.py`, and `routes_data.py`.

---

## 2. Energy integration (kWh)

**Formula:**
```
kWh = Σ (kW_i × dt_i)
```

where `dt_i` is the time elapsed from the previous sample to sample `i` (in hours).

**Integration method:** Left-point rectangular integration. Each reading is held constant from the previous sample forward to the current sample time.

**Code:** `metrics.py:compute_kpi()` lines 40-41, `behavior.py:compute_shift_energy_split()`, `trending.py:_daily_kwh()`

**Worked example (uniform cadence):**
```
Samples at 15-minute intervals (dt = 0.25 h):
  t=0:00  kW=50
  t=0:15  kW=60
  t=0:30  kW=40

kWh = (50 × 0.25) + (60 × 0.25) + (40 × 0.25)
    = 12.5 + 15.0 + 10.0
    = 37.5 kWh
```

Note: the first sample has `dt = 0` (no previous sample), so it contributes 0 kWh.

**Worked example (irregular cadence):**
```
  t=0:00  kW=50   dt=0.00 h  → 0.00 kWh
  t=0:15  kW=60   dt=0.25 h  → 15.00 kWh
  t=1:00  kW=40   dt=0.75 h  → 30.00 kWh
  t=1:15  kW=55   dt=0.25 h  → 13.75 kWh

Total = 58.75 kWh
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
