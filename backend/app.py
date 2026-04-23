import os
import socket
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PAGES_DIR = os.path.join(PROJECT_ROOT, "pages")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "captures")

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(PAGES_DIR, "index.html")


@app.route("/camera")
def camera():
    return send_from_directory(PAGES_DIR, "camera.html")


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
    os.makedirs(DATA_DIR, exist_ok=True)
    ip = get_local_ip()
    port = 5001
    print(f"\n  Local:   http://localhost:{port}")
    print(f"  Phone:   http://{ip}:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
