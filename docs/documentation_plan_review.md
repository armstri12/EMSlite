# EMSlite Documentation Plan — Review & Corrective Action Plan

_Date:_ 2026-02-20
_Reviewer:_ Claude (Opus 4.6)
_Scope:_ Review of documentation campaign plan, independent verification of Phase 0 audit findings, and corrective action recommendations.

## Executive Summary

The Codex Phase 0 audit identified 12 potential inconsistencies across calculation logic, data handling, and timezone semantics. **All 12 findings have been independently verified as accurate** against the source code. The documentation campaign plan is well-structured but has a critical gap: it treats these as documentation issues when several are actual bugs that need code fixes before they can be correctly documented.

---

## 1. Verification of Codex Audit Findings

Each finding was traced to exact source locations and assessed independently.

### Calculation Logic — 4/4 Confirmed

| # | Finding | Verdict | Evidence |
|---|---------|---------|----------|
| 1 | Unweighted avg_kw vs weighted total_kwh | **CONFIRMED** | `metrics.py:43` — `avg_kw = float(total_kw.mean())` is a simple arithmetic mean while `total_kwh` at line 41 uses `(total_kw * dt_hours).sum()`. Same pattern repeats in `compute_panel_rankings()` and `compute_department_breakdown()`. |
| 2 | fillna(0) masks missing telemetry | **CONFIRMED** | `metrics.py:32,79,121` and `routes_data.py:80,88` all use `amps_to_kw(df[col].fillna(0), ...)`. No distinction between "sensor offline" and "zero current". |
| 3 | Docstrings claim trapezoidal, code is rectangular | **CONFIRMED** | `behavior.py:119` docstring says "trapezoidal integration" but line 121-122 implements `kwh = kw * dt_hours` (left-point rectangular). Same mismatch in `trending.py:14-16`. |
| 4 | Panel vs facility percent-change mismatch | **CONFIRMED** | `trending.py:86` returns `100.0` when panel prior=0 and recent>0. `trending.py:138` returns `0.0` when facility prior=0. Different fallback policies for the same semantic operation. |

### Data Handling — 3/3 Confirmed

| # | Finding | Verdict | Evidence |
|---|---------|---------|----------|
| 5 | Department filter leaks full breakdown | **CONFIRMED** | `routes_data.py:140-144` filters `all_panels` by department for KPI, but lines 149-158 call `_get_all_department_panels()` and `compute_department_breakdown()` for every department unconditionally. |
| 6 | Total kW excludes displayed combo groups | **CONFIRMED** | `routes_data.py:60` — `meter_columns(df.columns, exclude=set(cfg.get("combo_columns", {}).keys()))` explicitly excludes combos. Lines 85-88 compute `total_kw_series` from this filtered set. Lines 91-97 compute `group_series` separately. Users see both but totals don't add up. |
| 7 | N+1 weather upsert | **CONFIRMED** | `weather.py:92-99` — `session.query(WeatherCache).filter(...).first()` runs inside a `for rec in records` loop. Classic N+1 pattern. |

### Timezone / Temporal — 5/5 Confirmed

| # | Finding | Verdict | Evidence |
|---|---------|---------|----------|
| 8 | Dual timezone representation | **CONFIRMED** | `ingest.py:139-143` writes as America/New_York with `dt.strftime("%Y-%m-%d %H:%M:%S%z")`. `core.py:54` reads back with `pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)`. Works via offset parsing but is fragile. |
| 9 | DST ambiguity drops records | **CONFIRMED** | `merge_meter_data.py:113` — `dt.dt.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")`. During fall-back, ambiguous times become NaT and get dropped at line 186: `df.dropna(subset=["Timestamp"])`. |
| 10 | Weather timestamp misinterpretation | **CONFIRMED** | `weather.py:51` — `datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)`. `.replace()` overwrites tzinfo without converting. A timestamp like `"2026-02-20T14:30:00-05:00"` would first parse correctly then have its offset silently replaced with UTC, shifting the time by 5 hours. |
| 11 | Fixed hourly count in cache heuristic | **CONFIRMED** | `weather.py:156-158` — `expected_hours = int((end_dt - start_dt).total_seconds() / 3600) + 1`. Assumes uniform hours. DST transitions and non-hourly source cadences make this inaccurate. |
| 12 | Naive DateTime columns with tz-aware values | **CONFIRMED** | `models.py:32,65,227,248` all declare `DateTime` without `timezone=True`. `weather.py:104-110` passes tz-aware UTC datetimes to `WeatherCache.timestamp`. SQLite stores these as naive text, stripping timezone info. |

**Score: 12/12 findings verified accurate.**

---

## 2. Assessment of the Documentation Campaign Plan

### Strengths

- **Logical phase structure.** Audit-first (Phase 0) before writing docs is correct. Foundation docs before detail docs prevents drift.
- **Defined quality bar.** The "module is documented when" criteria are specific and measurable.
- **Risk awareness.** Drift, over-commenting, and stale formulas are identified with practical mitigations.
- **Ownership model.** Cross-area review rotation reduces blind spots.
- **Correct priority ordering.** Starting with `metrics.py` and `core.py` addresses the highest-value surfaces first.

### Gaps and Weaknesses

1. **No code-fix track.** The plan is documentation-only. Three of the 12 findings are genuine bugs (weather timestamp misinterpretation, DST record dropping, misleading docstrings). Documenting incorrect behavior creates authoritative-looking wrong documentation — worse than no documentation.

2. **Time estimates assume documenting existing behavior.** Phase 2 is scoped at "2-4 days" for calculation semantics. With 4 calculation issues and 5 timezone issues to fix first, this is significantly underscoped.

3. **No testing component.** The QA phase covers doc linting and review, but doesn't include tests to validate corrected calculations. Without tests, fixes can regress silently.

4. **Weather module absent from priority ordering.** The priority list mentions `metrics.py`, `core.py`, route modules, `models.py`, `dashboard.js` — but not `weather.py`, which has 4 of the 12 issues.

5. **No data quality documentation deliverable.** The deliverables list `calculations.md`, `api/`, `data-model.md`, etc. but no section on telemetry assumptions, NaN semantics, or temporal invariants.

---

## 3. Severity Classification

### Critical — Data Correctness Risk

These produce silently wrong results and should be fixed before any documentation is written about the affected modules.

| # | Issue | Impact | Recommended Fix |
|---|-------|--------|----------------|
| 10 | `.replace(tzinfo=UTC)` in weather.py:51 | Shifts weather timestamps by up to hours | Change to `.astimezone(timezone.utc)` |
| 9 | `ambiguous='NaT'` in merge_meter_data.py:113 | Drops valid energy readings during DST fall-back | Use `ambiguous='infer'` or a keep-both strategy |
| 3 | Misleading "trapezoidal" docstrings | Maintainers trust docstrings for correctness decisions | Correct to "left-point rectangular" or implement actual trapezoidal |

### High — Misleading Results

These don't corrupt data but produce results that mislead users or create contradictions.

| # | Issue | Impact | Recommended Fix |
|---|-------|--------|----------------|
| 1 | Unweighted avg_kw (metrics.py:43) | Wrong averages with irregular sampling | Time-weighted mean: `(kw * dt).sum() / dt.sum()` |
| 2 | fillna(0) masks missing data | Understates consumption during sensor outages | Add NaN-handling strategy flag; document default behavior |
| 4 | Panel vs facility percent-change mismatch | Users see contradictory trend indicators | Unify to same fallback policy (recommend `null` when prior=0) |
| 8 | Dual timezone representation | Works today but fragile if any code path bypasses the shared loader | Document canonical standard; add assertion in `core.py` loader |

### Medium — Functional but Suboptimal

These work correctly for normal cases but have edge-case issues or performance concerns.

| # | Issue | Impact | Recommended Fix |
|---|-------|--------|----------------|
| 5 | Department filter returns full breakdown | Wasted computation; confusing API contract | Filter `dept_map` before calling `compute_department_breakdown()` |
| 6 | Total kW excludes combo groups | UI total doesn't match visible series | Document intent; optionally add `total_kw_including_groups` field |
| 7 | N+1 weather upsert | Performance degrades with large date ranges | Bulk-fetch existing timestamps, then batch insert/update |
| 11 | Fixed hourly count assumption | Cache completeness check inaccurate around DST | Account for DST transitions in expected-hours calculation |
| 12 | Naive DateTime columns | tz-aware values lose timezone on SQLite roundtrip | Use `DateTime(timezone=True)` or normalize to naive UTC before storage |

---

## 4. Corrective Action Plan

### Principle: Fix-Then-Document

Each documentation phase should **fix the code first**, then document the corrected behavior, then add tests to lock it in. This prevents the plan's biggest risk: creating authoritative documentation for buggy behavior.

### Revised Phase Breakdown

#### Phase 0 — Audit and Verification (COMPLETE)
- Codex audit: complete
- Independent verification: complete (this document)

#### Phase 1 — Critical Fixes + Foundation Docs (3-4 days)

**Code fixes:**
- [ ] `weather.py:51` — `.replace(tzinfo=UTC)` → `.astimezone(timezone.utc)`
- [ ] `merge_meter_data.py:113` — `ambiguous='NaT'` → `ambiguous='infer'`
- [ ] `behavior.py:119` and `trending.py:14` — correct docstrings to "left-point rectangular integration"
- [ ] `trending.py:86,138` — unify percent-change fallback to same policy for panel and facility

**Documentation:**
- [ ] `docs/README.md` — docs index and audience guide
- [ ] `docs/architecture.md` — system architecture and data flow overview
- [ ] `docs/temporal-assumptions.md` — canonical timezone standard, DST handling policy, boundary semantics

**Tests:**
- [ ] DST fall-back transition test for ingestion (verify records aren't dropped)
- [ ] Weather timestamp parsing test with offset inputs (verify correct UTC conversion)

#### Phase 2 — High-Priority Fixes + Calculation Docs (3-5 days)

**Code fixes:**
- [ ] `metrics.py:43` — implement time-weighted average: `(kw * dt).sum() / dt.sum()`
- [ ] Apply same fix to `compute_panel_rankings()` and `compute_department_breakdown()`
- [ ] Document and optionally flag the fillna(0) NaN-handling strategy
- [ ] Add timezone invariant assertion/comment in `core.py` loader

**Documentation:**
- [ ] `docs/calculations.md` — all formulas with units, assumptions, and worked examples
- [ ] `docs/data-quality.md` — NaN treatment, duplicate timestamps, irregular sampling implications

#### Phase 3 — Medium-Priority Fixes + API/Model Docs (3-5 days)

**Code fixes:**
- [ ] `routes_data.py` — filter `dept_map` to requested department in `/metrics` endpoint
- [ ] `routes_data.py` — document or resolve total_kw vs combo group relationship
- [ ] `weather.py` — replace N+1 upsert with bulk operation
- [ ] `weather.py` — fix cache hourly heuristic to account for DST
- [ ] `models.py` — add `timezone=True` to DateTime columns (evaluate migration impact)

**Documentation:**
- [ ] `docs/api/` — endpoint-by-endpoint references with request/response examples
- [ ] `docs/data-model.md` — entity relationships, lifecycle, and serialization notes

#### Phase 4 — Frontend + In-Code Docs (2-4 days)
- Same scope as original campaign Phases 4-5
- Dashboard lifecycle, chart helpers, state model
- Module-level and function docstrings pass
- Frontend comments for render pipelines

#### Phase 5 — QA and Governance (1-2 days)
- Same scope as original Phase 6, plus:
- [ ] Run full test suite — verify no regressions from code fixes
- [ ] Verify all 12 original findings are resolved (re-audit checklist)
- [ ] Establish PR-template doc-update checklist
- [ ] Set "last reviewed" metadata on each doc

---

## 5. Summary

| Aspect | Assessment |
|--------|-----------|
| Codex audit accuracy | **12/12** findings verified correct |
| Documentation plan quality | Good structure; missing code-fix integration |
| Biggest risk | Documenting buggy behavior as if correct (especially #10, #9, #3) |
| Key recommendation | Interleave code fixes with each documentation phase |
| Estimated total effort | ~15-20 days (up from 14-22 due to code fixes + tests) |

### Immediate Next Steps
1. Merge this review into the docs/ directory alongside the existing plan and findings.
2. Begin Phase 1 critical fixes: weather timestamp, DST handling, docstring corrections, percent-change unification.
3. Write foundation docs and temporal-assumptions document alongside the fixes.
