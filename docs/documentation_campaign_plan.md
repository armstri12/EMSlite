# EMSlite Documentation Campaign Plan

## Goal
Create complete, maintainable documentation coverage for EMSlite, including:
- Product behavior and user workflows
- API contracts and data model references
- Calculation logic and assumptions
- Implementation architecture and key module internals
- Inline comments/docstrings in complex code paths

## Scope
This campaign covers both backend and frontend surfaces:
- Python backend (`emslite/`): API routes, ingestion, metrics, weather, models, config, core conversion helpers
- Frontend (`emslite/static/`): dashboard architecture, state model, chart rendering patterns, and tab-specific logic
- Operational workflows: setup, ingestion lifecycle, troubleshooting, and deployment basics

## Deliverables

### 1) Documentation architecture
- `docs/README.md`: docs index and audience guide
- `docs/architecture.md`: system architecture and data flow
- `docs/calculations.md`: formulas, assumptions, worked examples, and edge cases
- `docs/api/`: endpoint-by-endpoint references
- `docs/data-model.md`: SQLAlchemy entity relationships and lifecycle notes
- `docs/frontend.md`: dashboard structure, state, rendering lifecycle
- `docs/operations.md`: local runbook, ingestion operations, backup/recovery notes
- `docs/contributing.md`: documentation and code-comment standards

### 2) In-code documentation improvements
- Module-level docstrings for each backend module lacking one
- Function docstrings for public helpers and non-trivial transforms
- Inline comments on domain-specific logic (e.g., amps→kW conversion, KPI rollups)
- Frontend comments for major render pipelines and shared Plotly helpers

### 3) Readme modernization
- Expand root `Readme.md` into a complete project entry point:
  - purpose, architecture snapshot, quick start, folder map, key links
  - pointers into `docs/` for deep dives

### 4) Verification artifacts
- Documentation coverage checklist by module
- "Last reviewed" dates and ownership metadata for each major doc
- Optional API schema export + link validation script

## Work Breakdown Structure

### Phase 0 — Audit and baseline (1–2 days)
1. Inventory modules, routes, models, and frontend entry points.
2. Capture undocumented calculations and implicit assumptions.
3. Build a coverage matrix: file/module → current docs status → required outputs.

### Phase 1 — Foundation docs (2–3 days)
1. Build `docs/README.md` and architecture overview.
2. Write operational quick start and environment prerequisites.
3. Expand root `Readme.md` and link to docs index.

### Phase 2 — Calculation and data semantics (2–4 days)
1. Document core formulas used by `core.py`, `metrics.py`, and ingestion transforms.
2. Include worked examples using realistic input rows.
3. Add "assumptions and caveats" sections for each KPI.

### Phase 3 — API and model references (3–5 days)
1. Document each route module (`routes_*.py`) with request/response examples.
2. Document entities and relationships from `models.py`.
3. Cross-link APIs to frontend usage points in `static/js/api.js` and `dashboard.js`.

### Phase 4 — Frontend implementation docs (2–4 days)
1. Document global state, tab render lifecycle, and chart helper abstractions.
2. Add component-level notes for each tab renderer.
3. Clarify extension points for new tabs/charts and performance considerations.

### Phase 5 — In-code comments/docstrings pass (3–6 days)
1. Add/normalize docstrings in backend modules.
2. Add targeted inline comments in complex logic paths only.
3. Avoid comment noise; prioritize "why" over "what".

### Phase 6 — QA and governance (1–2 days)
1. Tech review by maintainer(s) for correctness.
2. Run docs lint/link checks.
3. Establish ongoing doc maintenance process in PR template/contributing guide.

## Priority Ordering (recommended)
1. `emslite/metrics.py` and `emslite/core.py` (calculation truth source)
2. `emslite/api/routes_data.py`, `routes_config.py`, `routes_alerts.py`
3. `emslite/models.py` and `database.py`
4. `emslite/static/js/dashboard.js` and `api.js`
5. Remaining route modules + operations runbook

## Quality Bar
A module is considered documented when:
- Public behavior is explained in prose and examples
- Inputs/outputs and units are explicit
- Side effects and persistence interactions are called out
- Non-obvious algorithmic decisions are explained
- Cross-references to dependent modules are included

## Suggested ownership model
- **Documentation lead**: drives structure, consistency, and review cadence
- **Backend owner**: validates formulas, API docs, and model semantics
- **Frontend owner**: validates dashboard lifecycle and UX behavior
- **Reviewer rotation**: at least one reviewer outside the implementing area

## Risks and mitigations
- **Risk:** Drift between code and docs.
  - **Mitigation:** add doc update checklist to PR workflow and enforce in reviews.
- **Risk:** Over-commenting reduces readability.
  - **Mitigation:** style guide emphasizing intent-driven comments.
- **Risk:** Calculation docs become stale with config changes.
  - **Mitigation:** centralize formula references and note config dependencies.

## Success metrics
- 100% of backend modules have module-level docstrings
- 100% of externally consumed endpoints documented with examples
- 100% of KPI formulas documented with units and assumptions
- Root README supports a new developer quick start without external guidance
- Documentation review included in normal PR process

## Proposed execution rhythm
- Week 1: Phases 0–2
- Week 2: Phases 3–4
- Week 3: Phases 5–6 and governance hardening

## Immediate next actions
1. Approve this plan and appoint campaign owners.
2. Create tracking issue(s) from each phase and assign due dates.
3. Start with a metrics/core deep-dive to lock formula docs early.
