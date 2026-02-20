# GET /api/data

Returns time-series kW data for charting and dashboard rendering.

## Query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | ISO 8601 string | none | Filter start (inclusive) |
| `end` | ISO 8601 string | none | Filter end (inclusive) |
| `panels` | comma-separated string | all panels | Filter to specific panel IDs |
| `department` | string | none | Filter panels to this department |

## Response

```json
{
  "timestamps": ["2026-02-20T00:00:00Z", "2026-02-20T00:15:00Z", "..."],
  "total_kw": [83.2, 91.4, "..."],
  "panel_series": {
    "MeterATS01_SystemCurrent": [12.1, 13.5, "..."],
    "MeterATS02_SystemCurrent": [8.4, 9.1, "..."]
  },
  "panel_names": ["MeterATS01_SystemCurrent", "MeterATS02_SystemCurrent"],
  "group_series": {
    "Floor1_Total": [45.2, 48.3, "..."]
  },
  "group_names": ["Floor1_Total"],
  "rolling_hours": 1.0,
  "price_per_kwh": 0.25,
  "carbon_kg_per_kwh": 0.4
}
```

## Key notes

- **`total_kw`** is the sum of all physical panel columns only. `combo_columns` (pre-aggregated groups) are excluded to avoid double-counting.
- **`group_series`** contains pre-computed kW group columns from the CSV. These are already in kW units (not amps), so no conversion is applied.
- **`panel_series`** values are in kW, converted from amps via `amps × √3 × voltage × pf / 1000`.
- All arrays are the same length as `timestamps`. Missing panel readings are filled with `0.0`.
- Timestamps are UTC ISO 8601 strings with trailing `Z`.

## Example

```
GET /api/data?start=2026-02-20T08:00:00Z&end=2026-02-20T09:00:00Z&department=production
```
