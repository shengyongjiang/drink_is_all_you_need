import os
import socket
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

import config

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PAGES_DIR = os.path.join(PROJECT_ROOT, "pages")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "captures")

app = Flask(__name__, template_folder=PAGES_DIR)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(PAGES_DIR, "index.html")


@app.route("/camera")
def camera():
    return render_template("camera.html",
                           CAPTURE_INTERVAL=config.CAPTURE_INTERVAL,
                           ALERT_SECONDS=config.ALERT_SECONDS)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "no image"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = os.path.join(DATA_DIR, timestamp)
    os.makedirs(capture_dir, exist_ok=True)

    save_path = os.path.join(capture_dir, "original.jpg")
    file.save(save_path)

    return jsonify({"ok": True, "timestamp": timestamp})


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    from werkzeug.serving import make_server

    os.makedirs(DATA_DIR, exist_ok=True)
    ip = get_local_ip()
    http_port = 8010
    https_port = 8012

    print(f"\n  HTTP:     http://localhost:{http_port}/camera")
    print(f"  HTTP:     http://{ip}:{http_port}/camera")
    print(f"  HTTPS:    https://localhost:{https_port}/camera")
    print(f"  HTTPS:    https://{ip}:{https_port}/camera\n")

    http_server = make_server("0.0.0.0", http_port, app)
    threading.Thread(target=http_server.serve_forever, daemon=True).start()

    app.run(host="0.0.0.0", port=https_port, ssl_context="adhoc")
