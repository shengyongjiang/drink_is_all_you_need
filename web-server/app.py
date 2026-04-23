import json
import os
import socket
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

import config

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PAGES_DIR = os.path.join(os.path.dirname(__file__), "pages")
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


@app.route("/dashboard")
def dashboard():
    return send_from_directory(PAGES_DIR, "dashboard.html")


@app.route("/api/today")
def api_today():
    today_str = datetime.now().strftime("%Y%m%d")
    captures = []

    if os.path.isdir(DATA_DIR):
        for name in sorted(os.listdir(DATA_DIR)):
            if not name.startswith(today_str):
                continue
            result_path = os.path.join(DATA_DIR, name, "result.json")
            if not os.path.exists(result_path):
                continue
            with open(result_path) as f:
                data = json.load(f)
            if data.get("cup_detected"):
                data["_dir"] = name
                captures.append(data)

    captures.sort(key=lambda c: c["_dir"])

    events = []
    total_ml = 0.0
    for i, cap in enumerate(captures):
        ts = cap["_dir"]
        entry = {
            "timestamp": ts,
            "time": f"{ts[9:11]}:{ts[11:13]}",
            "level": cap["level"],
        }
        if i == 0:
            entry["type"] = "drink"
            entry["ml"] = 0
        else:
            prev_level = captures[i - 1]["level"]
            diff = prev_level - cap["level"]
            entry["level_before"] = prev_level
            entry["level_after"] = cap["level"]
            if diff > 0.01:
                entry["type"] = "drink"
                entry["ml"] = round(diff * config.CUP_CAPACITY_ML, 1)
                total_ml += entry["ml"]
            elif diff < -0.01:
                if prev_level <= 0.30:
                    ml = round(prev_level * config.CUP_CAPACITY_ML, 1)
                    entry["type"] = "refill"
                    entry["ml"] = ml
                    total_ml += ml
                else:
                    entry["type"] = "refill"
                    entry["ml"] = 0
            else:
                entry["type"] = "unchanged"
                entry["ml"] = 0
        events.append(entry)

    now = datetime.now()
    hours_elapsed = max(0, now.hour - config.ACTIVE_HOURS_START
                        + now.minute / 60)
    active_window = config.ACTIVE_HOURS_END - config.ACTIVE_HOURS_START
    expected_ml = round(
        (hours_elapsed / active_window) * config.DAILY_GOAL_ML, 1
    ) if hours_elapsed > 0 else 0

    total_ml = round(total_ml, 1)
    progress = round(min(total_ml / config.DAILY_GOAL_ML, 1.0), 4)

    status = "on_track"
    if total_ml < expected_ml * 0.8:
        status = "behind"
    elif total_ml >= config.DAILY_GOAL_ML:
        status = "completed"

    return jsonify({
        "date": now.strftime("%Y-%m-%d"),
        "cup_capacity_ml": config.CUP_CAPACITY_ML,
        "daily_goal_ml": config.DAILY_GOAL_ML,
        "notification_interval": config.NOTIFICATION_INTERVAL,
        "total_consumed_ml": total_ml,
        "goal_progress": progress,
        "events": events,
        "pace": {
            "expected_ml": expected_ml,
            "status": status,
        },
    })


@app.route("/api/capture/<timestamp>")
def api_capture(timestamp):
    capture_dir = os.path.join(DATA_DIR, timestamp)
    if not os.path.isdir(capture_dir):
        return jsonify({"error": "not found"}), 404

    files = sorted(os.listdir(capture_dir))
    images = [f for f in files if f.endswith((".jpg", ".png"))]

    result = {}
    result_path = os.path.join(capture_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)

    return jsonify({
        "timestamp": timestamp,
        "images": images,
        "result": result,
    })


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
