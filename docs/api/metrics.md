# GET /api/metrics

Returns computed KPI metrics, panel rankings, and department energy breakdown.

## Query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | ISO 8601 string | none | Filter start (inclusive) |
| `end` | ISO 8601 string | none | Filter end (inclusive) |
| `department` | string | none | Filter KPI and rankings to this department |

## Response

```json
{
  "kpi": {
    "total_kwh": 12450.5,
    "total_cost": 3112.63,
    "total_carbon_kg": 4980.2,
    "total_carbon_tonnes": 4.98,
    "peak_kw": 320.4,
    "avg_kw": 218.7,
    "load_factor": 68.3,
    "device_count": 12,
    "latest_timestamp": "2026-02-20T18:45:00Z",
    "date_range_days": 7.2
  },
  "rankings": [
    {
      "panel_id": "MeterATS01_SystemCurrent",
      "display_name": "Panel ATS-01",
      "total_kwh": 3241.2,
      "peak_kw": 95.4
    }
  ],
  "departments": [
    {
      "department": "Production",
      "total_kwh": 8200.0,
      "total_cost": 2050.0,
      "total_carbon_kg": 3280.0,
      "total_carbon_tonnes": 3.28,
      "peak_kw": 280.1,
      "device_count": 8
    }
  ],
  "carbon_kg_per_kwh": 0.4
}
```

## Department filter contract

When `?department=<id>` is provided:
- `kpi` and `rankings` reflect only the panels belonging to that department
- `departments` returns the breakdown for that department only (not all departments)

Without a department filter, `departments` returns all departments.

## KPI field definitions

| Field | Formula | Units |
|-------|---------|-------|
| `total_kwh` | Σ(kW × dt_hours) | kWh |
| `total_cost` | total_kwh × price_per_kwh | $ |
| `total_carbon_kg` | total_kwh × carbon_kg_per_kwh | kg CO₂ |
| `peak_kw` | max(total_kw_series) | kW |
| `avg_kw` | Σ(kW × dt) / Σ(dt) — time-weighted | kW |
| `load_factor` | avg_kw / peak_kw × 100 | % |

See [calculations.md](../calculations.md) for full formula documentation.
