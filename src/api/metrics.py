from __future__ import annotations

import time
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("http_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Total predictions", ["prediction"])

def now() -> float:
    return time.perf_counter()
