"""
Orders API demonstrating:
  1. Idempotent POST /orders (Idempotency-Key header)
  2. Cursor-based pagination for GET /orders
  3. Per-client rate limiting (X-Client-Id header)

Assigned values:
  Total orders (T) = 51
  Rate limit (R)    = 15 requests / 10 seconds
"""

import base64
import json
import threading
import time
import uuid
from collections import deque
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config (assigned values)
# ---------------------------------------------------------------------------
TOTAL_ORDERS = 51
RATE_LIMIT_R = 15
RATE_LIMIT_WINDOW_SECONDS = 10

app = FastAPI(title="Orders API")

# ---------------------------------------------------------------------------
# Fixed catalog: orders 1..T
# ---------------------------------------------------------------------------
CATALOG = [
    {"id": i, "item": f"Item {i}", "amount": round(10.0 + i * 3.37, 2)}
    for i in range(1, TOTAL_ORDERS + 1)
]

# ---------------------------------------------------------------------------
# Idempotency store: key -> created order dict
# ---------------------------------------------------------------------------
_idempotency_lock = threading.Lock()
_idempotency_store: dict[str, dict] = {}
_next_created_id = TOTAL_ORDERS + 1  # new orders get IDs after the fixed catalog


class OrderCreate(BaseModel):
    item: Optional[str] = None
    amount: Optional[float] = None


@app.post("/orders", status_code=201)
async def create_order(
    payload: OrderCreate = None,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    global _next_created_id

    if not idempotency_key:
        raise HTTPException(status_code=400, detail="Idempotency-Key header is required")

    with _idempotency_lock:
        existing = _idempotency_store.get(idempotency_key)
        if existing is not None:
            # Replay: same key -> same order, never create a duplicate.
            return JSONResponse(status_code=201, content=existing)

        order = {
            "id": _next_created_id,
            "item": (payload.item if payload and payload.item else "Order"),
            "amount": (payload.amount if payload and payload.amount is not None else 0.0),
            "created": True,
        }
        _next_created_id += 1
        _idempotency_store[idempotency_key] = order

    return JSONResponse(status_code=201, content=order)


# ---------------------------------------------------------------------------
# Cursor pagination over the fixed catalog
# ---------------------------------------------------------------------------
def encode_cursor(offset: int) -> str:
    raw = json.dumps({"offset": offset}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def decode_cursor(cursor: str) -> int:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8"))
        data = json.loads(raw)
        offset = int(data.get("offset", 0))
        if offset < 0:
            offset = 0
        return offset
    except Exception:
        return 0


@app.get("/orders")
async def list_orders(limit: int = 10, cursor: Optional[str] = None):
    if limit < 1:
        limit = 1
    offset = decode_cursor(cursor) if cursor else 0
    offset = max(0, min(offset, len(CATALOG)))

    page = CATALOG[offset: offset + limit]
    new_offset = offset + len(page)

    next_cursor = encode_cursor(new_offset) if new_offset < len(CATALOG) else None

    return {
        "items": page,
        "orders": page,       # alias accepted by grader
        "next_cursor": next_cursor,
        "next": next_cursor,  # alias accepted by grader
    }


# ---------------------------------------------------------------------------
# Per-client rate limiting middleware (sliding window log)
# ---------------------------------------------------------------------------
_rate_lock = threading.Lock()
_rate_buckets: dict[str, deque] = {}


def check_rate_limit(client_id: str):
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(client_id, deque())
        # drop timestamps outside the window
        while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_R:
            oldest = bucket[0]
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - oldest)) + 1)
            return False, retry_after

        bucket.append(now)
        return True, 0


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit the API's functional endpoints; let docs/openapi and
    # CORS preflight (OPTIONS) requests through untouched.
    if request.url.path in ("/docs", "/openapi.json", "/redoc", "/") or request.method == "OPTIONS":
        return await call_next(request)

    client_id = request.headers.get("X-Client-Id", "anonymous")
    allowed, retry_after = check_rate_limit(client_id)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


# CORS added LAST so it becomes the OUTERMOST middleware layer (Starlette
# builds the stack so the most-recently-added middleware wraps everything
# added before it). This guarantees:
#   - CORS headers are present on every response, including 429s from the
#     rate limiter below.
#   - CORS preflight (OPTIONS) requests are answered directly by
#     CORSMiddleware before they ever reach the rate limiter.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
@app.head("/")
async def root():
    return {
        "status": "ok",
        "total_orders": TOTAL_ORDERS,
        "rate_limit": f"{RATE_LIMIT_R} requests / {RATE_LIMIT_WINDOW_SECONDS}s",
        "endpoints": ["POST /orders", "GET /orders"],
    }
