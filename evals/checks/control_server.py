#!/usr/bin/env python3
"""A canned localcode supervisor (control server) for Checks: always 'ready'.
Usage: control_server.py <port> <model-port> <alias>"""
import http.server, json, sys
PORT, MPORT, ALIAS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        d = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/status": return self._json({"state": "ready", "model": ALIAS, "detail": "", "pct": None, "current": ALIAS, "port": MPORT, "ctx": 32768, "group": None, "filename": None, "vision": False, "vision_available": False, "vision_size_gb": 0.0})
        if p == "/voice/status": return self._json({"available": False})
        if p == "/progress": return self._json({})
        return self._json({"ok": True})
    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0); self.rfile.read(ln) if ln else None
        return self._json({"ok": True})
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
