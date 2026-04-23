# Trackin - Architecture

## Context

Water level detection system using AI vision pipeline (YOLO + SAM1 + SAM2). Decoupled architecture with three components communicating via a shared filesystem.

## Architecture

```
┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Component 1     │     │  Shared Folder        │     │  Component 2    │
│  Web Server      │────>│  data/captures/       │────>│  AI Processor   │
│  web-server/     │     │  {timestamp}/         │     │  ai/            │
│  (Flask + HTTPS) │     │    original.jpg       │     │  process.py     │
│                  │<────│    result.json        │<────│                 │
└──────────────────┘     └──────────────────────┘     └─────────────────┘
                                    │
                                    v
                         ┌─────────────────┐
                         │  Component 3    │
                         │  Frontend App   │
                         │  (DEFERRED)     │
                         └─────────────────┘
```

## Shared Folder Contract

**Location:** `data/captures/` (project root)

**Structure:** each capture gets its own timestamped folder
```
data/captures/
├── 20260423_143052/
│   ├── original.jpg        — uploaded photo
│   ├── yolo.jpg            — debug: YOLO detections
│   ├── cup.jpg             — debug: SAM cup mask
│   ├── regions.jpg         — debug: sub-region segmentation
│   ├── split.jpg           — debug: split score profile
│   ├── level.jpg           — debug: SAM1 water line overlay
│   ├── sam2fill.jpg        — debug: SAM2 overlay
│   └── result.json         — AI output
```

**Result JSON format:**
```json
{
  "timestamp": "20260423_143052",
  "processed_at": "2026-04-23T14:31:05",
  "cup_detected": true,
  "cup_class": "cup",
  "cup_confidence": 0.92,
  "cup_bbox": [120, 80, 450, 600],
  "sam1_level": 0.48,
  "sam2_level": 0.45,
  "level": 0.47
}
```

## Step 1: Web Server — DONE

### `web-server/app.py` — Flask Server

Serves camera page + receives uploads. Dual HTTP/HTTPS support.

**Routes:**
- `GET /` — serves `pages/index.html`
- `GET /camera` — serves camera page with config (Jinja2 template)
- `POST /upload` — receives photo, saves to `data/captures/{timestamp}/original.jpg`
- `GET /data/<path:filename>` — serves files from `data/captures/`

**Key details:**
- HTTP on port 8010, HTTPS on port 8012 (adhoc SSL, no PEM files needed)
- Bind to `0.0.0.0` so phone can reach it over local WiFi
- flask-cors enabled

### `web-server/pages/camera.html` — Mobile Camera Page

Live video stream with timed auto-capture.

**Features:**
- `getUserMedia` for live camera feed
- Configurable auto-capture interval (default 10min) with countdown alert
- Manual snap / cancel / auto-mode toggle
- Mobile-first CSS
- Config passed from server via Jinja2 (`CAPTURE_INTERVAL`, `ALERT_SECONDS`)

### `web-server/config.py` — Server Configuration

- `CAPTURE_INTERVAL` — seconds between auto-captures (default 600)
- `ALERT_SECONDS` — countdown warning before capture (default 10)

## Step 2: AI Processor — DONE

### `ai/process.py` — AI Processor

Refactored from `tests/test_pipeline.py`. Reusable module + CLI tool.

**Key function:**
```python
def process_image(capture_dir: str) -> dict
```

- Reads `original.jpg` from the capture directory
- Loads image (with EXIF rotation), resizes to 1024px
- Runs Approach 1: YOLO → SAM1 → OpenCV water level
- Runs Approach 2: SAM2 fine-tuned fill level
- Saves all debug images into the same directory
- Writes `result.json`
- Returns the result dict

**CLI usage:**
```bash
cd ai
uv run python process.py ../data/captures/20260423_143052/
uv run python process.py --all  # process all unprocessed captures
```

**AI modules:**
- `ai/vision.py` — YOLO detection, SAM segmentation, debug drawing
- `ai/volume_estimator.py` — water level detection from masks
- `ai/models/` — SAM1 checkpoint
- `ai/sam2_fill_level/` — SAM2 fine-tuned model
- `ai/tests/` — test scripts and test images

## Project Structure

```
hackathon_allyouneediswater/
├── web-server/                 # Component 1: Flask web server
│   ├── app.py                  # Flask app (HTTP 8010 + HTTPS 8012)
│   ├── config.py               # Capture interval / alert settings
│   ├── pages/
│   │   ├── index.html          # Landing page
│   │   ├── camera.html         # Live camera + auto-capture
│   │   └── dashboard.html      # React + Tailwind dashboard
│   └── pyproject.toml          # Flask + pyopenssl deps
├── ai/                         # Component 2: AI processor
│   ├── process.py              # Main processor (CLI + importable)
│   ├── vision.py               # YOLO + SAM detection
│   ├── volume_estimator.py     # Water level estimation
│   ├── models/                 # SAM1 checkpoint (gitignored)
│   ├── sam2_fill_level/        # SAM2 model (gitignored)
│   ├── tests/
│   │   ├── test_pipeline.py    # Batch test script
│   │   └── images/             # Test images
│   └── pyproject.toml          # AI deps (torch, ultralytics, etc.)
├── data/
│   └── captures/               # Shared folder (gitignored)
└── .gitignore
```

## Step 3: Frontend Dashboard — DONE

### `web-server/pages/dashboard.html` — React + Tailwind Dashboard

Single-page React app (CDN, no build step) showing daily water intake.

**Visual elements:**
- Large CSS water bottle with SVG wave animation and floating bubbles
- Graduation marks (0.5L–3.0L) on the side
- Water level fills up as user drinks, color shifts orange when behind pace
- Mood emoji indicator (🏜️→😐→😊→🎉)
- Progress bar showing pace status
- Stats cards (drink count, average ml, last drink time)
- Vertical timeline of drink events

**Tech:** React 18 + Tailwind CSS via CDN, Babel standalone for JSX

### `GET /api/today` — Daily Intake API

Server-side calculation of water consumption:
1. Scans `data/captures/` for today's directories with `result.json`
2. Compares consecutive `level` values — decreases > 0.01 = drinking events
3. `decrease × CUP_CAPACITY_ML` = ml consumed per drink
4. Returns: total consumed, goal progress, drink events, pace status

### `web-server/config.py` — New Settings

- `CUP_CAPACITY_ML = 300` — cup capacity in ml
- `DAILY_GOAL_ML = 3000` — daily water intake goal
- `ACTIVE_HOURS_START / END` — 8am–10pm active window
- `NOTIFICATION_INTERVAL = 15` — minutes between browser notifications

### Notification System

- Browser Notification API, permission requested via button
- Checks pace every 15 minutes during active hours
- Alerts when >20% behind expected intake
- Uses `tag: 'trackin-pace'` to prevent duplicate notifications

## Verification

1. `cd web-server && uv run python app.py` starts server
2. Open `http://<ip>:8010/camera` or `https://<ip>:8012/camera` on phone
3. Take photo → upload → saved to `data/captures/{timestamp}/original.jpg`
4. `cd ai && uv run python process.py --all` processes all unprocessed captures
5. `result.json` + debug images generated in each capture folder
6. Open `http://<ip>:8010/dashboard` to see daily water intake with bottle visualization
7. `/api/today` returns JSON with consumption data and pace status