"""Local end-to-end smoke test for the FastAPI app running on 127.0.0.1:8001."""
import csv
import io
import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"


def get_json(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=10).read())


def post_json(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def post_multipart_csv(path, rows):
    hdr = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr)
    for r in rows:
        w.writerow([r[h] for h in hdr])
    csv_bytes = buf.getvalue().encode()
    boundary = "----b" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="x.csv"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


cases = [
    ("Almaty",   dict(ROOMS=2, LONGITUDE=76.9286, LATITUDE=43.2567, TOTAL_AREA=58.0,
                       FLOOR=5, TOTAL_FLOORS=9,  FURNITURE=1, CONDITION=3,
                       CEILING=2.7, MATERIAL=2, YEAR=2010)),
    ("Astana",   dict(ROOMS=3, LONGITUDE=71.4491, LATITUDE=51.1694, TOTAL_AREA=85.0,
                       FLOOR=7, TOTAL_FLOORS=12, FURNITURE=1, CONDITION=3,
                       CEILING=2.8, MATERIAL=2, YEAR=2018)),
    ("Shymkent", dict(ROOMS=1, LONGITUDE=69.5900, LATITUDE=42.3417, TOTAL_AREA=42.0,
                       FLOOR=3, TOTAL_FLOORS=5,  FURNITURE=1, CONDITION=2,
                       CEILING=2.6, MATERIAL=1, YEAR=2005)),
    ("Aktobe",   dict(ROOMS=2, LONGITUDE=57.1669, LATITUDE=50.2839, TOTAL_AREA=52.0,
                       FLOOR=4, TOTAL_FLOORS=9,  FURNITURE=1, CONDITION=3,
                       CEILING=2.7, MATERIAL=2, YEAR=2012)),
    ("Atyrau",   dict(ROOMS=3, LONGITUDE=51.9236, LATITUDE=47.0945, TOTAL_AREA=74.0,
                       FLOOR=5, TOTAL_FLOORS=9,  FURNITURE=1, CONDITION=3,
                       CEILING=2.7, MATERIAL=2, YEAR=2015)),
]

print("=== /health ===")
print(" ", get_json("/health"))

print("\n=== /predict (single) ===")
for name, body in cases:
    res = post_json("/predict", body)
    ppsqm = res["price_kzt"] / body["TOTAL_AREA"]
    print(f"  {name:10s} total={res['price_kzt']:>14,.0f} KZT   per_sqm={ppsqm:>10,.0f} KZT/m2")

print("\n=== /batch (CSV, 5 rows) ===")
out = post_multipart_csv("/batch", [c[1] for c in cases])
print(f"  batch rows returned: {len(out)}")
keys = [k for k in out[0].keys() if k.startswith("pred") or k in ("price_kzt", "warning")]
print("  output keys sample:", list(out[0].keys())[:12])
for i, r in enumerate(out):
    pred = r.get("pred_price_kzt") or r.get("price_kzt") or 0
    print(f"    row{i}  pred={pred:>14,.0f}")

print("\n=== static assets ===")
for p in ("/", "/static/js/app.js", "/template/csv", "/template/xlsx"):
    try:
        resp = urllib.request.urlopen(BASE + p, timeout=10)
        print(f"  {p:25s} HTTP {resp.status}  size={len(resp.read())}")
    except Exception as exc:
        print(f"  {p:25s} FAIL: {exc}")

print("\nALL OK")
