# EMSlite Temporal Assumptions

This document defines the canonical timezone standard, DST handling policy, and timestamp boundary semantics for EMSlite. All contributors should treat this as the authoritative reference.

## Canonical timezone standard

**Rule: UTC at-rest, local only at display.**

| Layer | Timezone | Format |
|-------|---------|--------|
| Master CSV (at-rest) | America/New_York offset string | `2026-02-20 14:30:00-0500` |
| Analytics pipeline (in-memory) | UTC | `datetime64[ns, UTC]` |
| API responses | UTC ISO 8601 | `2026-02-20T19:30:00Z` |
| Frontend display | Browser local / Eastern | Formatted by JS |

The transition from CSV format to analytics format happens in `core.py:load_csv()`:

```python
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
```

`pd.to_datetime(..., utc=True)` correctly parses offset-aware strings (e.g., `-0500`) and converts them to UTC. This is safe because the offset is embedded in the string.

**Invariant**: Any code path that reads the master CSV **must** go through `core.py:load_csv()`. Bypassing this function risks operating on naive or incorrectly-zoned timestamps.

## Ingestion timezone handling

`emslite/ingest.py` receives raw panel CSVs whose timestamps are bare local strings (e.g., `18-Jan-26 3:15 AM EST`). The ingestion pipeline:

1. Strips `EST`/`EDT` suffix tokens (non-standard, not parseable as offset)
2. Parses the bare local string as America/New_York naive datetime
3. Localizes to `America/New_York` with:
   - `nonexistent="shift_forward"`: timestamps during spring-forward gaps are advanced to the first valid time after the gap
   - `ambiguous="infer"`: timestamps during fall-back overlaps are inferred from context (the surrounding sequence direction), preserving the record
4. Converts to offset string format for storage: `%Y-%m-%d %H:%M:%S%z`

This means the master CSV stores `America/New_York` offset strings. `load_csv()` then converts these to UTC. The round-trip is lossless because the offset is preserved in the stored string.

## DST transition behavior

### Spring-forward (second Sunday in March, 2:00 AM → 3:00 AM)
- The hour 2:00–2:59 AM Eastern does not exist
- `nonexistent="shift_forward"` advances these timestamps to 3:00 AM Eastern
- **Effect**: any reading timestamped in the gap gets shifted forward by up to 1 hour

### Fall-back (first Sunday in November, 2:00 AM → 1:00 AM)
- The hour 1:00–1:59 AM Eastern occurs twice
- `ambiguous="infer"` resolves ambiguity by using the direction of the surrounding sequence
- **Effect**: records during the overlap are preserved with the inferred UTC offset
- If the full series is ambiguous (no unambiguous context), pandas may still raise; the `except` fallback calls `tz_localize(TIMEZONE)` which may apply the standard (non-DST) offset

## Weather timestamp handling

NOAA NCEI API timestamps may include UTC offsets in ISO 8601 format. `emslite/weather.py` parses these correctly:

```python
parsed = datetime.fromisoformat(ts_str)
if parsed.tzinfo is None:
    ts = parsed.replace(tzinfo=timezone.utc)  # naive → assume UTC
else:
    ts = parsed.astimezone(timezone.utc)       # offset-aware → convert to UTC
```

The key distinction:
- `.replace(tzinfo=UTC)` — **overwrites** the timezone without converting. A string `2026-02-20T14:30:00-05:00` would be reinterpreted as 14:30 UTC instead of 19:30 UTC.
- `.astimezone(timezone.utc)` — **converts** correctly: `2026-02-20T14:30:00-05:00` → `2026-02-20T19:30:00+00:00`.

## Date filter boundary semantics

API endpoints accept `start` and `end` query parameters:

```python
start_dt = pd.to_datetime(start, utc=True)  # inclusive
end_dt = pd.to_datetime(end, utc=True)       # inclusive
df = df[(df["Timestamp"] >= start_dt) & (df["Timestamp"] <= end_dt)]
```

Both bounds are **inclusive**. Parameters should be ISO 8601 strings, optionally with timezone offset. Bare dates (e.g., `2026-02-20`) are interpreted as `2026-02-20T00:00:00Z`.

## Database DateTime columns

SQLAlchemy `DateTime` columns in `models.py` are declared without `timezone=True`. SQLite stores these as text and does not enforce timezone semantics. Callers that write tz-aware values should normalize to naive UTC before storage, or be aware that timezone info will be stripped on insertion.

The affected columns are:
- `Department.created_at`
- `Device.created_at`, `Device.updated_at`
- `FloorPlan.created_at`
- `AlertRule.created_at`
- `IngestLog.processed_at`
- `WeatherCache.timestamp`, `WeatherCache.fetched_at`
- `AlertEvent.event_ts`, `AlertEvent.acknowledged_at`

**Operational convention**: treat all values read back from these columns as naive UTC. Do not rely on SQLite to preserve or convert timezone information.

## Weather cache completeness heuristic

`weather.py:get_weather_for_range()` uses a heuristic to decide whether the cache is complete enough to skip an API fetch:

```python
expected_hours = int((end_dt - start_dt).total_seconds() / 3600) + 1
if len(cached) >= expected_hours * 0.8:
    return cached
```

**Limitation**: this assumes hourly cadence and uniform 24-hour days. DST transitions create 23-hour and 25-hour days which can make the check inaccurate by ±1 record. The 80% threshold provides tolerance for this and for sparse API data.

## Summary of invariants

| Invariant | Enforced by |
|-----------|------------|
| All analytics use UTC timestamps | `core.py:load_csv()` |
| Ingested timestamps preserve DST context | `ingest.py` + `merge_meter_data.py` |
| Weather timestamps are correctly converted (not replaced) | `weather.py:fetch_weather_from_api()` |
| API date filters use inclusive UTC bounds | `routes_data.py` |
| Database DateTime columns are effectively naive UTC | Convention (not enforced by schema) |
