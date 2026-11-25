"""Associate footage positions with actual location fixes near an event's video timestamp."""

import base64
import binascii
import json
import math
from dataclasses import dataclass

MAX_TRACE_HEADER = 6144
MAX_FIXES = 65
MAX_FIX_OFFSET_MS = 1500
MAX_ACCURACY_M = 50
STREET_ACCURACY_M = 25
PLACE_ACCURACY_M = 15


@dataclass(frozen=True)
class LocationFix:
    offset_ms: int
    latitude: float
    longitude: float
    accuracy_m: float


def read_trace(header: str | None) -> tuple[LocationFix, ...]:
    if header is None:
        return ()
    if len(header) > MAX_TRACE_HEADER:
        raise ValueError("location_trace_too_large")
    try:
        samples = json.loads(base64.b64decode(header, altchars=b"-_", validate=True))
        if not isinstance(samples, list) or len(samples) > MAX_FIXES:
            raise ValueError
        fixes = []
        previous = -1
        for sample in samples:
            if not isinstance(sample, list) or len(sample) != 4:
                raise ValueError
            offset, latitude, longitude, accuracy = sample
            if (
                type(offset) is not int
                or not previous < offset <= 60_000
                or not all(
                    type(value) in {float, int} and math.isfinite(value)
                    for value in (latitude, longitude, accuracy)
                )
                or not -90 <= latitude <= 90
                or not -180 <= longitude <= 180
                or not 0 < accuracy <= 10_000
            ):
                raise ValueError
            fixes.append(LocationFix(offset, latitude, longitude, accuracy))
            previous = offset
        return tuple(fixes)
    except (ValueError, UnicodeError, OverflowError, binascii.Error):
        raise ValueError("invalid_location_trace") from None


def event_location(timestamp: float | None, fixes: tuple[LocationFix, ...]) -> dict:
    if (
        type(timestamp) not in {int, float}
        or not math.isfinite(timestamp)
        or not 0 <= timestamp <= 60
    ):
        return {}
    event_ms = round(timestamp * 1000)
    closest = min(
        (
            fix
            for fix in fixes
            if fix.accuracy_m <= MAX_ACCURACY_M
            and abs(fix.offset_ms - event_ms) <= MAX_FIX_OFFSET_MS
        ),
        key=lambda fix: (abs(fix.offset_ms - event_ms), fix.accuracy_m),
        default=None,
    )
    if closest is None:
        return {}
    return {
        "location_available": True,
        "latitude": closest.latitude,
        "longitude": closest.longitude,
        "location_accuracy_m": closest.accuracy_m,
        "location_time_offset_ms": closest.offset_ms - event_ms,
        "location_source": "camera_device",
    }
