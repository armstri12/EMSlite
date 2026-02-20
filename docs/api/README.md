# EMSlite API Reference

_Base URL:_ `http://localhost:8000/api`

All endpoints are synchronous. All timestamps in responses are UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`). All timestamp query parameters accept ISO 8601 format with optional timezone offset.

## Endpoint index

| Method | Path | Description |
|--------|------|-------------|
| GET | [`/data`](data.md) | Time-series kW data for charting |
| GET | [`/metrics`](metrics.md) | KPI summary, panel rankings, department breakdown |
| GET | [`/config`](config.md) | Read system configuration |
| PUT | [`/config`](config.md) | Update system configuration |
| GET | `/devices` | List all devices/panels |
| POST | `/devices` | Create a device |
| GET | `/devices/{id}` | Get one device |
| PUT | `/devices/{id}` | Update a device |
| DELETE | `/devices/{id}` | Delete a device |
| GET | `/departments` | List all departments |
| POST | `/departments` | Create a department |
| GET | `/departments/{id}` | Get one department |
| PUT | `/departments/{id}` | Update a department |
| DELETE | `/departments/{id}` | Delete a department |
| GET | `/alerts/rules` | List alert rules |
| POST | `/alerts/rules` | Create alert rule |
| GET | `/alerts/events` | List alert events |
| GET | `/health` | Data health and completeness |
| GET | `/weather` | Cached weather data |
| GET | `/trending` | Per-panel trend snapshot |
| GET | `/trending/{panel_id}` | Single panel trend detail |
| GET | `/behavior/{panel_id}` | Shift/off-shift behavior analysis |
| GET | `/behavior/rankings` | Phantom draw rankings |

## Common query parameters

| Parameter | Format | Description |
|-----------|--------|-------------|
| `start` | ISO 8601 | Filter start (inclusive). Example: `2026-02-01T00:00:00Z` |
| `end` | ISO 8601 | Filter end (inclusive). Example: `2026-02-20T23:59:59Z` |
| `department` | string | Filter by department ID or display name |
