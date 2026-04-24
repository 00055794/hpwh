"""Comprehensive local test of the web app (127.0.0.1:8001).

Covers:
  /health                            - reachable + model_loaded=True
  /                                  - HTML ok
  static assets                      - CSS + JS
  /template/csv  /template/xlsx      - download
  /predict       (5 cities)          - single predictions, variance across lat/lon
  /predict       bad input 422       - FastAPI validation
  /batch         CSV 5 rows          - batch pipeline
  /batch         CSV with 1 bad row  - error reporting works
"""
from __future__ import annotations
import csv, io, json, sys, time, urllib.error, urllib.request, uuid

BASE = "http://127.0.0.1:8001"
OK, FAIL = "[OK]", "[FAIL]"
passed, failed = 0, 0


def rec(ok: bool, msg: str) -> None:
    global passed, failed
    if ok: passed += 1
    else:  failed += 1
    print(f"  {OK if ok else FAIL}  {msg}")


def get(path: str):
    t0 = time.time()
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return r.status, r.read(), int((time.time()-t0)*1000)


def post_json(path: str, body: dict):
    t0 = time.time()
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read()), int((time.time()-t0)*1000)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300], int((time.time()-t0)*1000)


def post_csv(path: str, rows: list[dict]):
    hdr = list(rows[0].keys())
    sb = io.StringIO(); w = csv.writer(sb); w.writerow(hdr)
    for r in rows: w.writerow([r.get(h, "") for h in hdr])
    b = "----b" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.csv\"\r\n"
            "Content-Type: text/csv\r\n\r\n").encode() + sb.getvalue().encode() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type":f"multipart/form-data; boundary={b}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read()), int((time.time()-t0)*1000)


def mk(longitude, latitude, **kw):
    d = dict(ROOMS=2, LONGITUDE=longitude, LATITUDE=latitude, TOTAL_AREA=58.0,
             FLOOR=5, TOTAL_FLOORS=9, FURNITURE=1, CONDITION=3,
             CEILING=2.7, MATERIAL=2, YEAR=2010)
    d.update(kw); return d


# ── 1. health ──
print("\n== 1. /health ==")
s, b, ms = get("/health")
j = json.loads(b)
rec(s == 200 and j.get("status") == "ok", f"HTTP {s}  body={j}  {ms}ms")
rec(j.get("model_loaded") is True, f"model_loaded={j.get('model_loaded')}")

# ── 2. index + static ──
print("\n== 2. index + static ==")
for p in ["/", "/static/js/app.js", "/static/css/style.css",
          "/template/csv", "/template/xlsx"]:
    try:
        s, b, ms = get(p)
        rec(s == 200 and len(b) > 100, f"{p:30s}  HTTP {s}  {len(b)}B  {ms}ms")
    except Exception as e:
        rec(False, f"{p:30s}  err={e}")

# ── 3. /predict single, five cities ──
print("\n== 3. /predict (5 cities) ==")
cities = [
    ("Almaty",     mk(76.9286, 43.2567)),
    ("Astana",     mk(71.4491, 51.1694)),
    ("Shymkent",   mk(69.5960, 42.3174, CONDITION=4, MATERIAL=3, YEAR=2015)),
    ("Aktobe",     mk(57.1670, 50.2797, TOTAL_AREA=52.0)),
    ("Atyrau",     mk(51.9238, 47.1167, TOTAL_AREA=74.0, YEAR=2018)),
]
preds = {}
for name, pay in cities:
    s, j, ms = post_json("/predict", pay)
    if s == 200 and isinstance(j, dict) and j.get("success") and "price_kzt" in j:
        preds[name] = j["price_kzt"]
        reg = (j.get("stat") or {}).get("Region", "?")
        rec(True, f"{name:10s}  total={int(j['price_kzt']):>14,} KZT   "
                  f"per_sqm={int(j['price_per_sqm']):>12,}   region={reg[:22]:22s}  {ms}ms")
    else:
        rec(False, f"{name}  HTTP {s}  resp={str(j)[:200]}")
# Variance sanity: predictions should differ across cities (not all identical)
vals = list(preds.values())
rec(len(set(vals)) >= 4, f"5 cities produced {len(set(vals))} distinct totals")

# ── 4. /predict invalid (expect 422) ──
print("\n== 4. /predict bad-input (422 expected) ==")
bad = mk(76.9286, 43.2567); bad["ROOMS"] = "not-a-number"
s, j, ms = post_json("/predict", bad)
rec(s in (400, 422), f"HTTP {s} {ms}ms  resp={str(j)[:120]}")

# ── 5. /batch 5-row CSV ──
print("\n== 5. /batch (5-row CSV) ==")
rows = [dict(pay, ROOMS=int(pay["ROOMS"])) for _, pay in cities]
s, j, ms = post_csv("/batch", rows)
rec(s == 200 and isinstance(j, list) and len(j) == 5,
    f"HTTP {s}  rows={len(j) if isinstance(j,list) else '?'}  {ms}ms")
has_pred = all("pred_price_per_sqm" in r and r["pred_price_per_sqm"] for r in j)
rec(has_pred, "every row has pred_price_per_sqm")
rec(not any("error" in r and r.get("error") for r in j), "no row error")

# ── 6. /batch with 1 bad row (expect error field, others OK) ──
print("\n== 6. /batch (1 row bad, 4 good) ==")
bad_rows = rows.copy()
bad_rows[2] = dict(rows[2]);  bad_rows[2]["LATITUDE"] = "bad"
try:
    s, j, ms = post_csv("/batch", bad_rows)
    rec(s == 200 and isinstance(j, list), f"HTTP {s}  rows={len(j)}  {ms}ms")
    errs = sum(1 for r in j if r.get("error"))
    rec(errs >= 1, f"{errs} row(s) reported with error")
    rec(sum(1 for r in j if r.get("pred_price_per_sqm")) == 4,
        f"{sum(1 for r in j if r.get('pred_price_per_sqm'))}/5 rows have predictions")
except urllib.error.HTTPError as e:
    rec(False, f"batch raised {e.code}")

# ── summary ──
print(f"\n== summary ==  passed={passed}  failed={failed}")
sys.exit(0 if failed == 0 else 1)
