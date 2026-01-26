# EMSlite

## Visualization script

Use `visualize_meter_data.py` to generate multiple interactive plots from a wide CSV export (Timestamp + meter columns).

### Example

```bash
python visualize_meter_data.py \
  --input RawPanelUsageHistory_UPDATED.csv \
  --output-dir visualizations \
  --top-n 6 \
  --rolling-window 1H
```

### Output files

The script writes HTML files to the output directory:

- `total_kw_timeseries.html`
- `total_kw_rolling_1h.html`
- `top_meters_timeseries.html`
- `total_kw_histogram.html`
- `daily_hour_heatmap.html`
