"""Regression tests for the energy-conversion fixes (over-reporting)."""

from __future__ import annotations

import pandas as pd
import pytest

from emslite.core import (
    amps_to_kw,
    build_meter_factor_maps,
    excluded_columns,
    meter_columns,
    meter_factors,
)
from emslite.metrics import compute_kpi, integrate_kwh

SQRT3 = 3 ** 0.5


def test_amps_to_kw_three_phase_formula():
    kw = amps_to_kw(pd.Series([100.0]), 480.0, 1.0, 1.0).iloc[0]
    assert kw == pytest.approx(100 * 480 * SQRT3 / 1000)


def test_power_factor_scales_real_power():
    base = amps_to_kw(pd.Series([100.0]), 480.0, 1.0, 1.0).iloc[0]
    assert amps_to_kw(pd.Series([100.0]), 480.0, 0.82, 1.0).iloc[0] == pytest.approx(0.82 * base)


def test_calibration_factor_scales_output():
    base = amps_to_kw(pd.Series([100.0]), 480.0, 1.0, 1.0).iloc[0]
    assert amps_to_kw(pd.Series([100.0]), 480.0, 1.0, 2.0).iloc[0] == pytest.approx(2 * base)
    # default keeps legacy behavior
    assert amps_to_kw(pd.Series([100.0]), 480.0, 1.0).iloc[0] == pytest.approx(base)


def test_integrate_kwh_trapezoidal_flat_load():
    ts = pd.date_range("2026-05-01", periods=5, freq="15min", tz="UTC")
    dt = pd.Series(ts).diff().dt.total_seconds().fillna(0) / 3600.0
    # Flat 10 kW over 1 h → 10 kWh (trapezoidal == right-endpoint for constant load)
    assert integrate_kwh(pd.Series([10.0] * 5), dt) == pytest.approx(10.0)


def test_integrate_kwh_trapezoidal_ramp():
    ts = pd.date_range("2026-05-01", periods=2, freq="1h", tz="UTC")
    dt = pd.Series(ts).diff().dt.total_seconds().fillna(0) / 3600.0
    # 0→10 kW over 1 h → trapezoid area = 5 kWh (right-endpoint would give 10)
    assert integrate_kwh(pd.Series([0.0, 10.0]), dt) == pytest.approx(5.0)


def test_compute_kpi_calibration_factor_scales_total():
    ts = pd.date_range("2026-05-01", periods=9, freq="15min", tz="UTC")
    df = pd.DataFrame({"Timestamp": ts, "P1": [100.0] * 9, "P2": [50.0] * 9})
    full = compute_kpi(df, 480.0, 0.82, panel_cols=["P1", "P2"])["total_kwh"]
    half = compute_kpi(df, 480.0, 0.82, panel_cols=["P1", "P2"], calibration_factor=0.5)["total_kwh"]
    assert half == pytest.approx(0.5 * full, rel=1e-6)


def test_excluded_columns_drops_aggregate_and_combo():
    cfg = {"combo_columns": {"Prod_kW": []}, "aggregate_columns": ["Main"]}
    assert excluded_columns(cfg) == {"Prod_kW", "Main"}
    cols = ["Timestamp", "Total_Amps", "Main", "P1", "P2", "Prod_kW"]
    assert meter_columns(cols, exclude=excluded_columns(cfg)) == ["P1", "P2"]


def test_meter_factors_overrides_with_fallback():
    cfg = {
        "power_factor": 1.0,
        "calibration_factor": 1.0,
        "meter_overrides": {
            "Meter A": {"calibration_factor": 1.2, "power_factor": 0.85},
            "Meter B": {"calibration_factor": 1.1},  # pf omitted → global
        },
    }
    assert meter_factors(cfg, "Meter A") == (0.85, 1.2)
    assert meter_factors(cfg, "Meter B") == (1.0, 1.1)
    assert meter_factors(cfg, "Unknown") == (1.0, 1.0)
    assert meter_factors(cfg, None) == (1.0, 1.0)


def test_build_meter_factor_maps_per_panel():
    cfg = {
        "power_factor": 1.0,
        "calibration_factor": 1.0,
        "meter_overrides": {
            "Meter A": {"calibration_factor": 1.2, "power_factor": 0.85},
            "Meter B": {"calibration_factor": 1.1},
        },
    }
    device_meter_map = {"P1": "Meter A", "P2": "Meter B", "P3": "Meter A"}
    pf_map, cal_map = build_meter_factor_maps(cfg, device_meter_map)
    assert cal_map == {"P1": 1.2, "P2": 1.1, "P3": 1.2}
    assert pf_map == {"P1": 0.85, "P3": 0.85}  # P2's meter has no PF override


def test_per_panel_calibration_in_compute_kpi():
    ts = pd.date_range("2026-05-01", periods=9, freq="15min", tz="UTC")
    df = pd.DataFrame({"Timestamp": ts, "P1": [100.0] * 9, "P2": [100.0] * 9})
    base = compute_kpi(df, 480.0, 1.0, panel_cols=["P1", "P2"])["total_kwh"]
    mixed = compute_kpi(df, 480.0, 1.0, panel_cols=["P1", "P2"], calibration_map={"P1": 2.0})["total_kwh"]
    # P1 doubled, P2 unchanged, equal panels → 1.5× the base total
    assert mixed == pytest.approx(1.5 * base, abs=0.01)
