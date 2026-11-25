#!/usr/bin/env python3

"""Serve the packaged results explorer with cross-origin isolation headers."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import webbrowser


HOST = "127.0.0.1"
PORT = 8000
EXPLORER_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "r"
    / "resultsExplorer"
    / "inst"
    / "www"
)


class ResultsExplorerHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


def main() -> None:
    os.chdir(EXPLORER_DIRECTORY)
    server = ThreadingHTTPServer((HOST, PORT), ResultsExplorerHandler)
    url = f"http://{HOST}:{PORT}/"
    print(f"Results explorer: {url}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
