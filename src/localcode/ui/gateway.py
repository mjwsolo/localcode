"""Movable loopback API listener; the authenticated inference server stays put.

Byte forwarding preserves SSE, request bodies and llama-server's authentication.
Retiring a listener stops new connections without closing established streams.
"""
from __future__ import annotations

import select
import socket
import socketserver
import threading
import time

from .ports import save_port


class PortChangeError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def relay(client: socket.socket, upstream: socket.socket, idle_timeout: float = 1800) -> None:
    """Bounded buffers and half-close support, including streaming responses."""
    sockets = (client, upstream)
    for sock in sockets:
        sock.setblocking(False)
    pending = {client: bytearray(), upstream: bytearray()}
    reading = {client, upstream}
    shut = set()
    deadline = time.monotonic() + idle_timeout
    while reading or any(pending.values()):
        for source, target in ((client, upstream), (upstream, client)):
            if source not in reading and not pending[target] and target not in shut:
                target.shutdown(socket.SHUT_WR)
                shut.add(target)
        readable = [source for source, target in ((client, upstream), (upstream, client))
                    if source in reading and len(pending[target]) < 65536]
        writable = [sock for sock in sockets if pending[sock]]
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("API connection idle timeout")
        ready, write, _ = select.select(readable, writable, [], min(timeout, 1.0))
        for source in ready:
            target = upstream if source is client else client
            try:
                chunk = source.recv(65536 - len(pending[target]))
            except BlockingIOError:
                continue
            if chunk:
                pending[target].extend(chunk)
                deadline = time.monotonic() + idle_timeout
            else:
                reading.remove(source)
        for target in write:
            try:
                count = target.send(pending[target])
            except BlockingIOError:
                continue
            if count == 0:
                raise ConnectionError("API connection closed")
            del pending[target][:count]
            deadline = time.monotonic() + idle_timeout


def unavailable(sock: socket.socket) -> None:
    try:
        sock.settimeout(1)
        body = b'{"error":{"message":"Model server unavailable; try again shortly","type":"server_error"}}'
        sock.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: "
                     + str(len(body)).encode() + b"\r\n\r\n" + body)
    except OSError:
        pass  # The client may already have disconnected.


class Listener(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, port: int, backend: int, capacity: threading.BoundedSemaphore):
        self.backend = backend
        self.capacity = capacity
        super().__init__(("127.0.0.1", port), Forward)

    def process_request(self, request, client_address):
        if not self.capacity.acquire(blocking=False):
            unavailable(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.capacity.release()
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.capacity.release()

    def start(self):
        thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        try:
            thread.start()
        except Exception:
            self.server_close()
            raise
        return self

    def retire(self):
        self.shutdown()
        self.server_close()  # Existing connection threads keep their own sockets.


class Forward(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            upstream = socket.create_connection(("127.0.0.1", self.server.backend), timeout=3)
        except OSError:
            unavailable(self.request)
            return
        with upstream:
            try:
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                relay(self.request, upstream)
            except (OSError, TimeoutError):
                # Never append JSON to an already-started SSE/HTTP response.
                # Closing the connection tells the client its stream ended.
                pass


class ModelGateway:
    def __init__(self, backend: int, port: int, control: int):
        self.backend, self.control = backend, control
        self._lock = threading.Lock()
        self._capacity = threading.BoundedSemaphore(64)
        self.listener = Listener(port, backend, self._capacity).start()
        self._closed = False

    @property
    def port(self):
        return self.listener.server_address[1]

    def change(self, port: int | None):
        if port is not None and (type(port) is not int or not 1024 <= port <= 65535):
            raise PortChangeError("Port must be an integer between 1024 and 65535, or auto", 400)
        if port in (self.backend, self.control):
            raise PortChangeError("That port is reserved for LocalCode's internal services")
        if not self._lock.acquire(blocking=False):
            raise PortChangeError("Another port change is in progress; refresh and try again")
        replacement = None
        try:
            if self._closed:
                raise PortChangeError("The model service is shutting down", 503)
            if port != self.port:
                try:
                    replacement = Listener(port or 0, self.backend, self._capacity).start()
                except OSError as exc:
                    raise PortChangeError(f"Cannot use port {port or 'auto'}: {exc.strerror or str(exc)}. The current port is unchanged.") from exc
                except RuntimeError as exc:
                    raise PortChangeError("Could not start the new listener. The current port is unchanged.", 503) from exc
            try:
                save_port(port)
            except OSError as exc:
                raise PortChangeError("Could not save the port setting. The current port is unchanged.", 500) from exc
            if replacement is not None:
                previous, self.listener = self.listener, replacement
                replacement = None
                previous.retire()
            return {"ok": True, "port": self.port, "base_url": f"http://127.0.0.1:{self.port}/v1", "restart_required": False}
        finally:
            try:
                if replacement is not None:
                    replacement.retire()
            finally:
                self._lock.release()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.listener.retire()
