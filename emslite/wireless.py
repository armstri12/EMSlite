"""Monnit Alta wireless sensor TCP ingestion.

Listens for JSON-over-TCP pushes from a Monnit Alta Ethernet Gateway,
auto-registers gateways and sensors, and stores readings in the database.

Message format: newline-delimited JSON, each object containing:
  gatewayMessage: {gatewayID, gatewayName, networkID, ...}
  sensorMessages: [{sensorID, sensorName, dataType, dataValue,
                    messageDate, signalStrength, batteryLevel}, ...]

Timestamps arrive as Microsoft /Date(ms)/ format and are converted to UTC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Module-level state shared with the routes layer (read via get_status()).
_state: dict = {
    "running": False,
    "port": None,
    "last_message_at": None,
    "connected_gateways": 0,
    "server": None,
    "auto_discover": True,
}

# Monnit dataType integer → sensor type name.
# Unmapped integers are stored as "dataType_N" for later configuration.
_DATATYPE_MAP: dict[int, str] = {
    1: "temperature",
    2: "humidity",
    3: "ambient_light",
    4: "voltage_ac",
    5: "voltage_dc",
    6: "current_ac",
    7: "current_dc",
    8: "door_window",
    9: "motion",
    10: "vibration",
    11: "water",
    12: "button",
    13: "thermocouple",
    14: "differential_pressure",
    15: "activity",
    20: "co2",
    21: "air_quality",
}


def _map_sensor_type(data_type: int) -> str:
    return _DATATYPE_MAP.get(data_type, f"dataType_{data_type}")


def _parse_monnit_date(date_str: str) -> datetime:
    """Parse Monnit /Date(ms)/ or /Date(ms+0500)/ to a UTC datetime."""
    m = re.search(r"\d+", date_str)
    if not m:
        raise ValueError(f"Cannot parse Monnit date: {date_str!r}")
    return datetime.fromtimestamp(int(m.group()) / 1000.0, tz=timezone.utc)


# ── Database helpers (synchronous, called from async context via thread pool) ──

def _ensure_gateway(gateway_id: str, display_name: str) -> None:
    from .database import get_session
    from .models import WirelessGateway

    session = get_session()
    try:
        if session.get(WirelessGateway, gateway_id) is None:
            session.add(WirelessGateway(id=gateway_id, display_name=display_name, enabled=True))
            session.commit()
            logger.info("Auto-registered gateway %s (%s)", gateway_id, display_name)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _touch_gateway(gateway_id: str, when: datetime) -> None:
    from .database import get_session
    from .models import WirelessGateway

    session = get_session()
    try:
        gw = session.get(WirelessGateway, gateway_id)
        if gw:
            gw.last_seen = when
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def _ensure_sensor(sensor_id: str, gateway_id: str, display_name: str, data_type: int) -> None:
    from .database import get_session
    from .models import WirelessSensor

    session = get_session()
    try:
        if session.get(WirelessSensor, sensor_id) is None:
            sensor_type = _map_sensor_type(data_type)
            session.add(WirelessSensor(
                id=sensor_id,
                gateway_id=gateway_id,
                display_name=display_name,
                sensor_type=sensor_type,
                enabled=True,
            ))
            session.commit()
            logger.info("Auto-registered sensor %s (%s) type=%s", sensor_id, display_name, sensor_type)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _store_reading(
    sensor_id: str,
    timestamp: datetime,
    value: float | None,
    signal_strength: int | None,
    battery_level: int | None,
) -> None:
    from .database import get_session
    from .models import SensorReading

    session = get_session()
    try:
        session.add(SensorReading(
            sensor_id=sensor_id,
            timestamp=timestamp,
            value=value,
            signal_strength=signal_strength,
            battery_level=battery_level,
        ))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Message processing ──

def _process_message(raw_json: str) -> None:
    """Parse a complete JSON push from the gateway and persist all readings."""
    try:
        msg = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON from gateway: %s — %.200s", exc, raw_json)
        return

    gw_msg = msg.get("gatewayMessage") or {}
    gateway_id = str(gw_msg.get("gatewayID") or gw_msg.get("networkID") or "unknown")
    gateway_name = gw_msg.get("gatewayName") or f"Gateway {gateway_id}"

    _ensure_gateway(gateway_id, gateway_name)
    now = datetime.now(tz=timezone.utc)
    _touch_gateway(gateway_id, now)

    for smsg in msg.get("sensorMessages") or []:
        sensor_id = str(smsg.get("sensorID", "")).strip()
        if not sensor_id:
            continue

        sensor_name = smsg.get("sensorName") or f"Sensor {sensor_id}"

        try:
            data_type = int(smsg.get("dataType", 0))
        except (TypeError, ValueError):
            data_type = 0

        date_str = smsg.get("messageDate") or ""
        try:
            ts = _parse_monnit_date(date_str) if date_str else now
        except ValueError:
            ts = now
            logger.warning("Could not parse messageDate %r for sensor %s", date_str, sensor_id)

        raw_val = smsg.get("dataValue")
        try:
            value = float(raw_val) if raw_val is not None else None
        except (TypeError, ValueError):
            value = None

        sig_raw = smsg.get("signalStrength")
        sig = int(sig_raw) if sig_raw is not None else None
        bat_raw = smsg.get("batteryLevel")
        bat = int(bat_raw) if bat_raw is not None else None

        if _state.get("auto_discover", True):
            _ensure_sensor(sensor_id, gateway_id, sensor_name, data_type)

        _store_reading(sensor_id, ts, value, sig, bat)
        logger.debug("Stored reading sensor=%s ts=%s val=%s sig=%s bat=%s",
                     sensor_id, ts.isoformat(), value, sig, bat)

    _state["last_message_at"] = now


# ── Async TCP connection handler ──

async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a single gateway TCP connection. Reads newline-framed JSON."""
    peer = writer.get_extra_info("peername")
    logger.info("Wireless gateway connected from %s", peer)
    _state["connected_gateways"] += 1
    buf = b""
    loop = asyncio.get_event_loop()
    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                raw = line.strip()
                if raw:
                    # Run synchronous DB calls in thread pool to avoid blocking the event loop.
                    await loop.run_in_executor(
                        None, _process_message, raw.decode("utf-8", errors="replace")
                    )
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("Error reading from gateway %s: %s", peer, exc)
    finally:
        _state["connected_gateways"] = max(0, _state["connected_gateways"] - 1)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("Wireless gateway disconnected from %s", peer)


# ── Public lifecycle functions called from app.py lifespan ──

async def start_listener(port: int, auto_discover: bool = True) -> None:
    """Start the asyncio TCP server on the FastAPI event loop."""
    _state["auto_discover"] = auto_discover
    _state["port"] = port
    server = await asyncio.start_server(_handle_connection, host="0.0.0.0", port=port)
    _state["server"] = server
    _state["running"] = True
    _state["connected_gateways"] = 0
    logger.info("Wireless TCP listener started on port %d (auto_discover=%s)", port, auto_discover)


async def stop_listener() -> None:
    """Stop the TCP server gracefully."""
    server = _state.get("server")
    if server is not None:
        server.close()
        await server.wait_closed()
    _state["running"] = False
    _state["server"] = None
    logger.info("Wireless TCP listener stopped.")


def get_status() -> dict:
    """Return current listener state (safe to call from synchronous route handlers)."""
    last = _state.get("last_message_at")
    return {
        "running": _state.get("running", False),
        "port": _state.get("port"),
        "last_message_at": last.isoformat() if last else None,
        "connected_gateways": _state.get("connected_gateways", 0),
        "auto_discover": _state.get("auto_discover", True),
    }
