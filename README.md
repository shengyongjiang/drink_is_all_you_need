# Trackin — AI Water Intake Tracker

A water intake tracking system that uses AI vision (YOLO + SAM) to detect water levels in cups from phone photos, calculates daily consumption, and visualizes progress on an animated dashboard.

## Architecture

```
┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Web Server      │     │  Shared Folder        │     │  AI Processor   │
│  web-server/     │────>│  data/captures/       │────>│  ai/            │
│  Flask + HTTPS   │     │  {timestamp}/         │     │  YOLO + SAM1/2  │
│                  │<────│  original.jpg         │<────│                 │
│  /camera         │     │  result.json          │     │  process.py     │
│  /dashboard      │     │  debug images         │     │                 │
└──────────────────┘     └──────────────────────┘     └─────────────────┘
```

- **Web Server** — Flask app serving a mobile camera page and a React + Tailwind dashboard
- **AI Processor** — YOLO detects the cup, SAM segments it, OpenCV + SAM2 estimate water level
- **Shared Folder** — `data/captures/{timestamp}/` with uploaded photos, AI results, and debug images

## Quick Start

### 1. Start the web server

```bash
cd web-server
uv run python app.py
```

Runs on HTTP `:8010` and HTTPS `:8012` (adhoc SSL).

### 2. Capture photos

Open `http://<your-ip>:8010/camera` on your phone. The camera page supports manual snap and timed auto-capture.

### 3. Run AI processing

```bash
cd ai
uv run python process.py --all
```

Processes all unprocessed captures and writes `result.json` + debug images into each capture folder.

### 4. View the dashboard

Open `http://<your-ip>:8010/dashboard` to see:

- Animated water bottle showing remaining intake goal (3L default)
- Green pace line indicating expected consumption
- Timeline of all captures with drink/refill/no-change events
- Click any event to view AI analysis images and detection details
- Browser notifications when behind drinking pace

## How It Works

1. Phone camera captures a photo of the cup → uploaded to `data/captures/{timestamp}/original.jpg`
2. AI pipeline detects the cup (YOLO), segments it (SAM), and estimates water level (0–100%)
3. Dashboard compares consecutive captures:
   - **Level decreased** → water was consumed (ml = decrease × cup capacity)
   - **Level increased** → cup was refilled
   - **Refill after ≤30%** → assumes remaining water was finished before refill
4. Daily consumption is summed and displayed against the 3L goal

## Configuration

Edit `web-server/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CUP_CAPACITY_ML` | 300 | Cup capacity in ml |
| `DAILY_GOAL_ML` | 3000 | Daily water intake goal |
| `CAPTURE_INTERVAL` | 600 | Auto-capture interval (seconds) |
| `ACTIVE_HOURS_START/END` | 8 / 22 | Active hours for pace tracking |
| `NOTIFICATION_INTERVAL` | 15 | Minutes between browser notifications |

## Project Structure

```
├── web-server/
│   ├── app.py                 # Flask app + API endpoints
│   ├── config.py              # Server and tracking configuration
│   └── pages/
│       ├── index.html         # Landing page
│       ├── camera.html        # Mobile camera + auto-capture
│       └── dashboard.html     # React + Tailwind dashboard
├── ai/
│   ├── process.py             # AI processing pipeline (CLI + module)
│   ├── vision.py              # YOLO + SAM detection
│   ├── volume_estimator.py    # Water level estimation
│   ├── models/                # SAM1 checkpoint (gitignored)
│   ├── sam2_fill_level/       # SAM2 fine-tuned model (gitignored)
│   └── tests/                 # Test images and scripts
└── data/
    └── captures/              # Shared capture folder (gitignored)
```

## Dependencies

- **Web Server:** Flask, flask-cors, pyopenssl
- **AI Processor:** PyTorch, ultralytics (YOLO), segment-anything (SAM), OpenCV
- **Dashboard:** React 18, Tailwind CSS (via CDN, no build step)

All Python dependencies managed via `uv` with per-component `pyproject.toml`.
