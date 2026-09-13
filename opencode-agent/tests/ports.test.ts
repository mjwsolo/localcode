import { expect, test } from "bun:test"
import path from "node:path"

test("an existing picker prevents a second launch, while unrelated listeners reserve ports", () => {
  const result = Bun.spawnSync(["python3", "-c", `
import json, socket, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ports import choose_ports
with socket.socket() as future, socket.socket() as busy, socket.socket() as free:
    future.bind(('127.0.0.1', 0))
    reserved = future.getsockname()[1]
    future.close()
    busy.bind(('127.0.0.1', 0))
    occupied = busy.getsockname()[1]
    free.bind(('127.0.0.1', 0))
    unused = free.getsockname()[1]
    free.close()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'state': 'idle', 'port': reserved}).encode())
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as control:
        control.bind(('127.0.0.1', 0))
        next_control = control.getsockname()[1]
    try:
        try:
            choose_ports([reserved, occupied, unused], [server.server_port, next_control])
            raise AssertionError('second launch accepted')
        except RuntimeError as error:
            assert 'already running' in str(error)
        assert choose_ports([occupied, unused], [next_control]) == (unused, next_control)
    finally:
        server.shutdown()
        server.server_close()
`], { cwd: path.resolve(import.meta.dir, "..") })
  expect(result.stderr.toString()).toBe("")
  expect(result.exitCode).toBe(0)
})
