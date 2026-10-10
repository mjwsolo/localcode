"""Loopback fixture servers need no hostname discovery or external DNS."""
from http.server import ThreadingHTTPServer
from socketserver import TCPServer


class LoopbackHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind calls getfqdn(), which can block on hosted Macs.
        # Our clients always use the numeric loopback address.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
