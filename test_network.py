# test_network.py
# Deploy lên Render free Web Service để test outbound network
# Sau khi test xong -> xóa service này

from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import json
import os

TESTS = [
    ("Fireant API",    "https://restv2.fireant.vn/posts?symbol=VCB&limit=3"),
    ("Fireant Web",    "https://fireant.vn/symbol/VCB"),
    ("f319 Main",      "https://f319.com"),
    ("f319 Search",    "https://f319.com/search?q=VCB"),
    ("CafeF RSS",      "https://cafef.vn/rss/thi-truong-chung-khoan.rss"),
    ("VnExpress RSS",  "https://vnexpress.net/rss/kinh-doanh.rss"),
    ("Entrade API",    "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from=1700000000&to=1800000000&symbol=VCB&resolution=D"),
    ("DeepSeek API",   "https://api.deepseek.com/v1/models"),
    ("Google News",    "https://news.google.com/rss/search?q=VCB+chung+khoan&hl=vi&gl=VN"),
]

def run_tests():
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}
    for name, url in TESTS:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            deny = r.headers.get("x-deny-reason", "")
            status = r.status_code
            ok = status == 200
            note = deny if deny else (r.text[:60] if not ok else "OK")
            results.append({"name": name, "status": status, "ok": ok, "note": note})
        except Exception as e:
            results.append({"name": name, "status": 0, "ok": False, "note": str(e)[:80]})
    return results

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/test":
            results = run_tests()
            body = "<h2>Render Network Test</h2><table border=1>"
            body += "<tr><th>Source</th><th>Status</th><th>OK?</th><th>Note</th></tr>"
            for r in results:
                color = "green" if r["ok"] else "red"
                body += f"<tr><td>{r['name']}</td><td>{r['status']}</td>"
                body += f"<td style='color:{color}'>{'YES' if r['ok'] else 'NO'}</td>"
                body += f"<td>{r['note']}</td></tr>"
            body += "</table>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK - go to /test to run network tests")

    def log_message(self, format, *args):
        pass  # tắt log spam

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Test server running on port {port}")
    print(f"Visit: https://your-app.onrender.com/test")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
