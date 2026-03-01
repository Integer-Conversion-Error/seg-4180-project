#!/usr/bin/env python3
"""
Simple HTTP server for point_manager.html
"""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent.absolute()
TARGET_FILE = "point_manager.html"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def translate_path(self, path):
        path = path.split('?')[0]
        if path == '/':
            path = '/' + TARGET_FILE
        return super().translate_path(path)
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    target_path = DIRECTORY / TARGET_FILE
    if not target_path.exists():
        print(f"ERROR: {TARGET_FILE} not found in {DIRECTORY}")
        exit(1)
    
    print(f"Serving {TARGET_FILE} on http://localhost:{PORT}")
    print("Press Ctrl+C to stop")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.shutdown()
