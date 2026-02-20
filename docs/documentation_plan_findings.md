# EMSlite Documentation Plan — Phase 0 Audit Findings

_Date:_ 2026-02-20  
_Scope reviewed:_ calculations, data handling, timezone behavior, and documentation-driven enhancement opportunities.

## Method
This audit followed the campaign plan's Phase 0 intent (inventory and assumptions capture) and focused on the highest-priority calculation and API surfaces first (`core.py`, `metrics.py`, ingestion, behavior/trending, and key routes).

## Potential inconsistencies

### 1) Calculation logic
1. **Energy integration uses interval deltas, but average kW is unweighted.**  
   `total_kwh` is interval-weighted (`kw * dt_hours`) while `avg_kw` is a simple arithmetic mean over rows. With irregular sample cadence, these can represent different realities and may appear inconsistent to users.
2. **Missing panel readings are treated as zero load in KPI/ranking math.**  
   Multiple paths use `fillna(0)` before conversion to kW; this can understate consumption when NaN means "missing telemetry" instead of true zero current.
3. **Behavior module claims trapezoidal integration but implements single-point multiplication.**  
   The docstring says trapezoidal, but the implementation is `kwh = kw * dt_hours` (not midpoint/trapezoid), which may mislead maintainers and documentation readers.
4. **Trending facility-level percent-change fallback differs from per-panel fallback.**  
   Panels return `100%` when prior is zero and recent is positive, while facility summary returns `0%` if prior total is zero; this creates interpretation mismatch.

### 2) Data handling
1. **Department-filtered metrics still return full department breakdown map.**  
   `/metrics` filters KPI/rankings by `department`, but department breakdown is always computed for all departments, potentially conflicting with a filtered page context.
2. **Total kW excludes configured combo/group columns while groups are returned separately.**  
   This can produce perceived mismatch between visible grouped series and total series, especially if users expect totals to include everything shown.
3. **Potentially expensive weather upsert pattern.**  
   Weather cache writes perform a query-per-record upsert loop; this is functionally correct but can degrade with long date ranges.

### 3) Time zone and temporal semantics
1. **Ingestion persists master timestamps in America/New_York string format, while analytics normalize reads to UTC.**  
   This can be valid, but dual representations increase risk of subtle boundary issues if any path bypasses the shared loader.
2. **Timestamp parser strips `EST/EDT` then localizes with DST ambiguity handling that can drop ambiguous records (`ambiguous='NaT'`).**  
   Around fall-back transitions, valid readings may become NaT and be dropped.
3. **Weather API timestamp handling force-replaces timezone with UTC.**  
   `datetime.fromisoformat(...).replace(tzinfo=UTC)` can reinterpret already-offset timestamps incorrectly instead of converting them.
4. **Weather cache completeness heuristic assumes fixed hourly count from UTC date bounds.**  
   DST/non-uniform source cadence can make expected-hours checks inaccurate.
5. **Database DateTime columns are declared without timezone-aware column types while code frequently passes tz-aware values.**  
   In SQLite this may serialize as naive text and rehydrate inconsistently across call paths.

## Potential enhancements

### High priority
1. **Define and document a canonical time standard (UTC at-rest, local only at display).** Add explicit invariants in `docs/calculations.md` and route docs.
2. **Introduce a shared energy integration utility** (e.g., `integrate_kw_to_kwh(series, ts, method='left'|'trapezoid')`) and use it in metrics/behavior/trending for consistency.
3. **Differentiate missing vs zero telemetry.** Add an option/flag for NaN handling strategy and document defaults.
4. **Normalize percent-change semantics** across panel and facility trend summaries (same prior=0 policy).

### Medium priority
5. **Clarify `/metrics?department=` contract** (filtered-only response vs global+filtered sections).
6. **Document `total_kw` vs `group_series` intent** and optionally expose `total_kw_including_groups` when that interpretation is needed.
7. **Optimize weather cache writes** with bulk upsert/merge strategy.
8. **Add DST-focused test fixtures** for ingestion parsing and behavior shift classification.

### Governance / documentation quality
9. **Create a "temporal assumptions" section** in docs for: timezone conversions, DST handling, inclusive/exclusive window boundaries, and date filter behavior.
10. **Add explicit worked examples for irregular sampling cadence** showing why weighted average and simple mean can diverge.
11. **Add data quality notes** describing treatment of NaN, duplicate timestamps, and inferred values.
12. **Track formula ownership and last-reviewed metadata** per the campaign plan to reduce drift risk.

## Suggested next execution step
Move into Phase 2 draft docs by turning the top four high-priority items above into:
- one calculation semantics page,
- one temporal handling page,
- one API contract clarification table.
