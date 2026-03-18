# EMSlite Calculation Sets — Slide Deck Source Notes

_Last reviewed: 2026-03-17_

This document is a presentation-ready breakdown of **every major calculation set** in EMSlite: what is calculated, the formula, why it exists, where it is used, and caveats to communicate clearly in slides.

---

## 1) Foundational electrical conversion

### 1.1 Amps → kW (three-phase)

**Formula**

\[
\mathrm{kW} = \mathrm{amps} \times \frac{\mathrm{voltage} \times \sqrt{3} \times \mathrm{power\_factor}}{1000}
\]

**Why this exists**
- Raw panel telemetry is current (amps), but all portfolio KPIs are in power/energy/cost units.
- This is the canonical conversion reused throughout backend and frontend workflows.

**Inputs**
- `amps`: sampled current reading per device.
- `line_voltage`: config default 480V; can be overridden per device.
- `power_factor`: config default 1.0.

**Where it is used**
- Core helper (`amps_to_kw`) and then consumed by metrics, behavior, trending, alerts, bills, and API data shaping.

**Caveats for slides**
- Assumes balanced three-phase behavior.
- Power factor errors propagate into every downstream metric.

---

## 2) Time integration and core KPI math

### 2.1 Elapsed time per sample

**Formula**

\[
\Delta t_i = \frac{\mathrm{Timestamp}_i - \mathrm{Timestamp}_{i-1}}{3600\ \mathrm{s/hr}}
\]

with first interval set to `0`.

### 2.2 Energy integration (kWh)

**Formula**

\[
\mathrm{total\_kWh} = \sum_i \left(\mathrm{kW}_i \times \Delta t_i\right)
\]

**Method**
- Left-point rectangular integration on sampled points.
- Irregular sampling cadence is supported because each point is weighted by elapsed time.

### 2.3 Average demand (time-weighted)

**Formula**

\[
\mathrm{avg\_kW} = \frac{\sum_i (\mathrm{kW}_i \times \Delta t_i)}{\sum_i \Delta t_i}
\]

fallback to simple mean when total elapsed time is zero.

### 2.4 Peak demand

**Formula**

\[
\mathrm{peak\_kW} = \max_i(\mathrm{kW}_i)
\]

### 2.5 Load factor

**Formula**

\[
\mathrm{load\_factor}(\%) = \frac{\mathrm{avg\_kW}}{\mathrm{peak\_kW}} \times 100
\]

### 2.6 Cost and carbon conversion

**Formulas**

\[
\mathrm{total\_cost} = \mathrm{total\_kWh} \times \mathrm{price\_per\_kWh}
\]
\[
\mathrm{carbon\_kg} = \mathrm{total\_kWh} \times \mathrm{carbon\_kg\_per\_kWh}
\]
\[
\mathrm{carbon\_tonnes} = \frac{\mathrm{carbon\_kg}}{1000}
\]

**Why this exists**
- Provides executive/business framing of electrical usage.

---

## 3) Ranking and allocation calculation sets

### 3.1 Panel rankings

**Formula set per panel**
- Convert panel amps to kW.
- Integrate to panel kWh over selected period.
- Compute panel peak kW.
- Sort descending by panel kWh and return top N.

### 3.2 Department breakdown

**Formula set per department**
- Sum kW across all mapped valid panels at each timestamp.
- Integrate summed kW to department kWh.
- Compute cost/carbon with configured factors.
- Compute department peak kW and device count.
- Sort departments by total kWh.

### 3.3 YTD allocation percentages

**Formulas**

\[
\%\mathrm{cost\_share} = \frac{\mathrm{dept\_cost}}{\mathrm{facility\_cost}} \times 100
\]
\[
\%\mathrm{kWh\_share} = \frac{\mathrm{dept\_kWh}}{\mathrm{facility\_kWh}} \times 100
\]

### 3.4 Monthly department mix

**Formula**

\[
\%\mathrm{monthly\_dept\_cost} = \frac{\mathrm{dept\_cost\_month}}{\mathrm{facility\_cost\_month}} \times 100
\]

---

## 4) Trending and change detection sets

### 4.1 Window partitioning

For a requested lookback window (default 7 days):
- `recent`: latest timestamp minus window duration through latest timestamp.
- `prior`: immediately preceding equal-duration window.

### 4.2 Period-over-period change

**Formula**

\[
\%\Delta = \frac{\mathrm{recent\_kWh} - \mathrm{prior\_kWh}}{\mathrm{prior\_kWh}} \times 100
\]

**Zero-baseline handling**
- `prior > 0`: normal formula.
- `prior = 0` and `recent > 0`: treated as `+100%`.
- `prior = 0` and `recent = 0`: `0%`.

### 4.3 Trend labels

- Rising: `pct_change > +5%`
- Falling: `pct_change < -5%`
- Stable: otherwise

### 4.4 Rolling trend smoothing

- Daily kWh series is smoothed with a 7-day rolling average for detail views.
- Point-level kW chart uses rolling-mean smoothing with window size inferred from average sample interval (targeting ~24h-equivalent points).

### 4.5 Hour-of-day and weekday profiles

- Group by hour (0–23) or weekday and compute arithmetic mean kW for each bucket.
- Used to visually identify recurring load-shape patterns.

---

## 5) Behavior / phantom-load calculation sets

### 5.1 Shift classification

**Rule set**
- Shift = weekdays, 6:00 to 14:30 America/New_York.
- Off-shift = all other times.

### 5.2 Shift vs off-shift energy split

**Formula set**
- Compute `kWh_i = kW_i * dt_i`.
- Sum by class (shift vs off-shift).
- Percent split:

\[
\mathrm{shift\_pct} = \frac{\mathrm{shift\_kWh}}{\mathrm{total\_kWh}} \times 100
\]

(and equivalently for off-shift).

### 5.3 Phantom draw estimate

**Deep off-hour filter**
- Weeknights, 23:00–03:59 Eastern.

**Primary formula**

\[
\mathrm{phantom\_kW} = P_{25}(\mathrm{kW\ in\ deep\ off-hours})
\]

Also reports mean, median, and sample count in that filtered segment.

**Rationale**
- 25th percentile is robust against outliers and transient outages versus using minimum.

### 5.4 Annualized phantom impact

**Constants and formulas**

\[
\mathrm{off\_shift\_hours/year} = (15.5 \times 260) + (24 \times 105) = 6550
\]
\[
\mathrm{annual\_phantom\_kWh} = \mathrm{phantom\_kW} \times 6550
\]
\[
\mathrm{annual\_phantom\_cost} = \mathrm{annual\_phantom\_kWh} \times \mathrm{price\_per\_kWh}
\]
\[
\mathrm{annual\_phantom\_carbon\_kg} = \mathrm{annual\_phantom\_kWh} \times \mathrm{carbon\_kg\_per\_kWh}
\]

### 5.5 Reduction scenarios (25/50/75/100%)

For each reduction level `r`:

\[
\mathrm{kW\_saved} = \mathrm{phantom\_kW} \times r
\]
\[
\mathrm{kWh\_saved/year} = \mathrm{kW\_saved} \times 6550
\]
\[
\mathrm{cost\_saved/year} = \mathrm{kWh\_saved/year} \times \mathrm{price\_per\_kWh}
\]
\[
\mathrm{carbon\_saved/year} = \mathrm{kWh\_saved/year} \times \mathrm{carbon\_kg\_per\_kWh}
\]

---

## 6) Alerts calculation set

### 6.1 Alert windowing and threshold checks

For each configured and enabled device:
1. Convert amps to kW.
2. For each sample in window:
   - critical if `kW >= critical_kw`
   - else warning if `kW >= warning_kw`
3. Emit event payload with current kW, threshold, amps, timestamp, and severity.

### 6.2 Alert ordering and summary

- Sort order priority: critical > warning, unacknowledged > acknowledged, then newest timestamp.
- Summary counts are simple cardinalities by severity.

---

## 7) Billing reconciliation calculation set

For a selected utility bill period and meter assignment:
1. Find enabled devices mapped to the meter.
2. Filter telemetry to bill date range.
3. Per device, convert amps→kW and integrate to kWh.
4. Sum device kWh to meter-period kWh.
5. Compute expected metered cost:

\[
\mathrm{calculated\_cost} = \mathrm{total\_kWh} \times \mathrm{price\_per\_kWh}
\]

6. Reconciliation delta:

\[
\mathrm{difference} = \mathrm{bill\_amount} - \mathrm{calculated\_cost}
\]

**Slide rationale**
- This is a sanity/variance check between billed and telemetry-derived economics.

---

## 8) API response shaping calculations (`/api/data`)

### 8.1 Facility `total_kw`

- At each timestamp, `total_kw` is the sum of **physical meter columns only**.
- Configured combo/group columns are excluded from this sum to prevent double-counting.

### 8.2 Group/combo series

- Each configured group is emitted separately as the row-wise sum of its component panel columns.
- These are analysis overlays; they are not added into facility `total_kw`.

---

## 9) Frontend dashboard calculation set (client-side derivations)

### 9.1 Core helper metrics in browser

Frontend computes its own derived windows for cards/charts:
- `windowKwh`: integrates series over selected window via \(\sum kW_i \times dt_i\).
- `windowAvgKw`: arithmetic average of included points.
- `windowPeakKw`: max kW in window.

### 9.2 Period summaries and comparisons

Derived for facility and department/group cards:
- Last full week kWh/cost.
- Prior week kWh/cost.
- Last weekend vs prior weekend.
- MTD usage/cost.
- Percent deltas computed as \((a-b)/b\times100\) when baseline is non-zero.

### 9.3 Department/top-consumer aggregations

- For each department/group, sums window kWh and average kW across constituent series.
- Top consumers are ranked by window kWh and monetized with `cost = kWh * price`.

**Important caveat to call out**
- Some frontend averages are arithmetic means over samples, while backend KPI average is time-weighted. This can diverge under irregular sampling.

---

## 10) Executive report synthesis calculations

### 10.1 Week-over-week deltas

For each KPI category (`kWh`, `cost`, `carbon`, `peak`, `avg`, `load_factor`):

\[
\Delta = \mathrm{current} - \mathrm{prior}
\]
\[
\Delta\% = \frac{\Delta}{\mathrm{prior}} \times 100 \quad (\mathrm{if\ prior}>0;\ else\ 0)
\]

### 10.2 Four-week baseline normalization

- Compute 4-week totals, convert to weekly average by dividing by number of weeks in window.
- Compare this-week metrics against that weekly baseline using same delta formulas.

### 10.3 Significant panel movements

- From trending snapshot, panels with `abs(pct_change) > 10` are considered significant.
- Cost impact inherits trending panel `cost_change`.

### 10.4 YTD YoY comparison

For each department:

\[
\mathrm{YoY\ cost\ delta} = \mathrm{cost\_this\_YTD} - \mathrm{cost\_prior\_YTD\_window}
\]
\[
\mathrm{YoY\ cost\ pct} = \frac{\mathrm{YoY\ cost\ delta}}{\mathrm{cost\_prior\_YTD\_window}} \times 100
\]

(with zero-baseline guard).

---

## 11) Recommended slide structure (ready-to-build)

1. **Telemetry to Value Chain** — amps → kW → kWh → cost/carbon.
2. **KPI Engine** — integration, weighted average, peak, load factor.
3. **Allocation Engine** — panel ranking + department contribution math.
4. **Trend Engine** — recent/prior windows, percent change, smoothing.
5. **Behavior Engine** — shift split + phantom baseline + annualized savings.
6. **Operations Controls** — alerts and bill reconciliation logic.
7. **Frontend vs Backend nuance** — arithmetic vs weighted averages.
8. **Assumptions & Limitations** — three-phase assumption, power factor dependency, zero-baseline policy, sampling cadence sensitivity.

---

## 12) Assumptions and rationale to explicitly state in presentation

- The platform is designed around current-based meter telemetry, not direct real power metering.
- Economic and carbon outputs are linear transforms of integrated energy and configured factors.
- Zero-baseline periods use explicit fallback policies to avoid divide-by-zero errors.
- Aggregation safeguards intentionally prevent combo/group double counting.
- Behavioral analytics prioritize robust baseline estimation (percentiles) over minima.


---

## 13) Suggested future improvements (roadmap-ready slide content)

Use this section as a “What to improve next” chapter in your deck. It ties directly to the current calculation design.

### 13.1 Increase sample resolution and quality controls

**What to improve**
- Increase telemetry cadence where practical (e.g., from 15-minute to 5-minute or 1-minute intervals for critical feeders).
- Add ingestion quality flags for missing intervals, duplicates, and stale samples.

**Why it matters**
- Finer sampling captures short-lived peaks and cycling equipment behavior that coarse intervals can hide.
- Better time resolution improves demand diagnostics, event attribution, and confidence in interval-derived kWh.

**Implementation notes**
- Preserve current time-weighted integration (`kW * dt`) so mixed/irregular cadence still computes correctly.
- Add data-quality KPIs in reports (coverage %, expected vs observed intervals, longest gap).

### 13.2 Add true three-phase real-power sensing

**What to improve**
- Add meters/sensors that capture per-phase voltage, current, and real power (kW) directly for all three phases.
- Where possible, capture `kW`, `kVAR`, and `kVA` instead of relying on assumed power factor.

**Why it matters**
- Current conversion assumes balanced three-phase conditions and fixed PF; direct real-power metering removes this approximation.
- Improves accuracy for unbalanced loads, variable PF equipment, and mixed electrical characteristics.

**Implementation notes**
- Add a source-priority rule: prefer measured kW when available, otherwise fall back to amps-converted kW.
- Version the methodology in reports so stakeholders know which meters are measured vs inferred.

### 13.3 Move from static to dynamic emissions factors

**What to improve**
- Replace single carbon factor with time/location-aware marginal or grid-intensity factors.

**Why it matters**
- Carbon impact becomes more decision-useful (same kWh can imply different emissions by hour/season).

**Implementation notes**
- Store emission factor as a timeseries and compute `carbon_kg_i = kWh_i * factor_i` at interval level.

### 13.4 Add tariff-aware cost modeling (beyond flat $/kWh)

**What to improve**
- Support time-of-use energy rates, demand charges, ratchets, and seasonal tariff windows.

**Why it matters**
- Aligns analytics to real utility economics and strengthens bill reconciliation/forecasting.

**Implementation notes**
- Calculate energy cost per interval with applicable tariff block.
- Add monthly demand charge estimator using billing-interval peak demand.

### 13.5 Harmonize frontend/backend metric semantics

**What to improve**
- Standardize average kW and percent-change logic between frontend and backend (single source of truth).

**Why it matters**
- Avoids user confusion when card values differ between API-backed reports and browser-derived widgets.

**Implementation notes**
- Prefer server-computed metrics for business-critical cards.
- Publish metric definitions in API contracts and test fixtures.

### 13.6 Strengthen baseline and anomaly analytics

**What to improve**
- Expand phantom/baseline analytics using seasonal baselines and weather normalization.
- Add robust anomaly detection (e.g., z-score on residuals, change-point detection) for early fault detection.

**Why it matters**
- Separates normal operational variance from actionable inefficiencies.

### 13.7 Improve metadata and organizational mapping integrity

**What to improve**
- Enforce meter-to-device and department mapping validation rules (no orphaned/overlapping assignments unless explicitly allowed).

**Why it matters**
- Allocation and attribution quality depends on clean hierarchy/mapping data.

### 13.8 Add uncertainty bands for inferred values

**What to improve**
- Add confidence or uncertainty indicators for inferred kW/kWh (especially when PF or voltage are assumed).

**Why it matters**
- Helps executives interpret precision and prioritize instrumentation upgrades where uncertainty is highest.

### 13.9 Operational rollout priorities (suggested order)

1. **Data quality + cadence improvements** (highest immediate lift).
2. **Three-phase true-power instrumentation** for top feeders.
3. **Tariff-aware cost engine** and enhanced bill reconciliation.
4. **Dynamic carbon factors**.
5. **Advanced anomaly + forecasting layer**.

---

## 14) Explicit end-to-end data flow (where data moves, transforms, and persists)

This section is intentionally implementation-specific so you can present the **actual shuttle path** of EMSlite data.

### 14.1 Runtime bootstrap and wiring

1. API process starts and loads `visualization_config.json` into in-memory app config.
2. SQLite database `emslite.db` is initialized/migrated.
3. Filesystem directories are ensured:
   - drops folder (default `drops/`) for inbound meter files
   - data folder (default `data/`) for the master merged CSV
4. Existing master CSV is scanned to sync known device IDs.
5. Any already-present drop files are ingested.
6. File watcher is started to continuously ingest new files.

**Presentation takeaway:** EMSlite operates as a hybrid file + DB pipeline: high-volume timeseries in CSV, metadata/state in SQLite.

### 14.2 Raw telemetry ingestion path (file stream)

**Inbound stream**
- Source files: `Meter*_SystemCurrent.csv` (configurable glob), dropped into `drops/`.

**Ingest steps per file**
1. Parse panel CSV into a two-column shape (`Timestamp`, meter column).
2. Auto-discover meter/device ID and upsert into `devices` metadata table if new.
3. Read current master CSV (`data/RawPanelUsageHistory.csv` by default).
4. Merge by timestamp:
   - fill missing values for existing timestamps
   - append truly new timestamps
5. Sort by timestamp and write back to master CSV.
6. Move source file to `drops/processed/` on success or `drops/failed/` on exception.
7. Write ingestion audit row into `ingest_logs` table.

**Presentation takeaway:** The master CSV is the canonical telemetry store used for all downstream calculations.

### 14.3 Weather ingestion path (optional API stream)

**Inbound stream**
- External weather API fetch for configured station.

**Persisted form**
- Cached in `weather_cache` table as timestamped temperature/humidity rows.

**Downstream usage**
- `/api/weather` returns weather rows for chart overlays and analysis context.

### 14.4 Storage architecture and what lives where

| Store | Location | Data classes stored | Primary readers/writers |
|---|---|---|---|
| Config JSON | `visualization_config.json` | rates, voltage/PF defaults, weather settings, combo groups | app bootstrap, `/api/config` |
| Master telemetry CSV | `data/RawPanelUsageHistory.csv` (default) | wide timeseries (`Timestamp` + one column per device) | ingest writer; `/api/data`, metrics, trending, behavior, alerts, reports, bills readers |
| Drop folders | `drops/`, `drops/processed/`, `drops/failed/` | inbound raw files + ingest outcomes | ingest watcher |
| SQLite DB | `emslite.db` | devices, departments, rules, floorplans, ingest logs, weather cache, bills, alert ack state | all API routes as needed |

### 14.5 Core calculation shuttle (telemetry → KPIs)

1. API route loads master CSV into dataframe.
2. Resolve meter columns (excluding synthetic totals and configured combo columns where appropriate).
3. Convert amps→kW per column (device voltage override first, else global line voltage; global PF).
4. Aggregate as needed:
   - row-wise sum for facility `total_kw`
   - per-group sums for combo series
   - per-department sums from department→device map
5. Build `dt_hours = Timestamp.diff()/3600`.
6. Integrate `kWh = Σ(kW * dt_hours)` for any requested window/scope.
7. Apply derivative calculations (avg kW, peak, load factor, cost, carbon, pct change, rankings).
8. Serialize to API response objects consumed by frontend cards/charts.

### 14.6 API data streams available to frontend (contract view)

- `/api/data`:
  - `timestamps`
  - `total_kw` (physical-meter sum only)
  - `panel_series` (per-device kW series)
  - `group_series` and `group_names` (configured combo/group overlays)
- `/api/metrics`: facility KPI block + top panels + department breakdown
- `/api/trending/snapshot`, `/api/trending/detail`: period comparisons, rolling series, profiles
- `/api/behavior`, `/api/behavior/rankings`: shift split, phantom estimates, reduction scenarios
- `/api/alerts`: threshold exceedance event stream + summary counts
- `/api/weather`: cached temperature/humidity streams
- `/api/reports/*`: pre-assembled weekly/YTD report data payloads
- `/api/bills/*`: CRUD bill records + telemetry-vs-bill comparison stream
- `/api/devices`, `/api/departments`, `/api/config`, `/api/ingest-log`: metadata and governance streams

### 14.7 Frontend transformation layer (client-side manipulations)

After fetching API payloads, the dashboard performs additional local derivations:
- windowed kWh integration for selected date spans
- arithmetic avg and peak computations for cards
- period boundary slicing (last full week, prior week, weekend windows, MTD)
- department/group and top-consumer ranking summaries

**Important:** these browser-side derivations can diverge from backend KPI semantics if formulas differ (e.g., arithmetic vs time-weighted averages).

### 14.8 Stateful vs stateless data pathways

- **Stateful/persisted**: master telemetry CSV, SQLite metadata tables, alert acknowledgements, weather cache, utility bills, ingest audit trail.
- **Stateless/computed on read**: KPI aggregates, trend deltas, profiles, report rollups.

### 14.9 Data lineage map for one KPI example (`total_cost`)

1. Raw amps sample arrives via drop CSV.
2. Ingestion merges sample into master CSV row/column cell.
3. Metric route loads row range and converts amps→kW.
4. Integrates to `total_kWh` using elapsed-time weighting.
5. Applies configured `price_per_kwh` scalar.
6. Emits `total_cost` in `/api/metrics` and report payloads.

This is a clean “raw telemetry → normalized power → integrated energy → business metric” lineage.
