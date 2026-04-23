---
theme: default
title: "Trackin - AI Water Intake Tracker"
info: |
  Hackathon Project - All You Need Is Water
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

# 💧 Trackin

### *~~Attention~~ Water Is All You Need*

<br>

 AI-Powered Water Intake Tracker

<br>
<br>

<div class="abs-br m-6 text-sm opacity-50">
Hackathon 2026
</div>

---
transition: fade-out
---

# The Problem

<br>

<div class="grid grid-cols-2 gap-8">

<div>

### 😵 We Forget to Drink Water

- Most people don't drink enough water daily
- Busy work schedules make it easy to forget

<br>
<br>

### The Idea

- Track water intake **automatically**
- No manual input required
- Smart reminders when you fall behind
</div>

<div>

[//]: # (<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/3046730_32596.jpg" class="w-full rounded-lg" />)
<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/Screenshot 2026-04-23 at 3.11.13 PM.png" class="w-full rounded-lg" />

</div>

</div>

---

# Our Solution

<br>

<div class="text-center text-xl mb-8">

A camera watches your cup. AI detects how much you drink. That's it.

</div>

<div class="grid grid-cols-3 gap-6 text-center">

<div class="p-4 bg-blue-50 rounded-lg">

### 📸
### Auto Capture
Photo every 10 min

</div>

<div class="p-4 bg-blue-50 rounded-lg">

### 🤖
### AI Analysis
Detect water level

</div>

<div class="p-4 bg-blue-50 rounded-lg">

### 📊
### Dashboard
Track your progress

</div>

</div>

---

# How It Works

<br>

```
📸 Camera ──→ 🖼️ Image Upload ──→ 🤖 AI Pipeline ──→ 📊 Dashboard
   (auto)        (every 10min)      (YOLO + SAM)     (real-time)
```

<br>

<div class="grid grid-cols-2 gap-8">

<div>

### AI Pipeline

1. **YOLO** - Detect the cup in the image
2. **SAM + SAM2** - Segment the cup precisely
3. **Level Detection** - Calculate water level (0-100%)

</div>

<div>

### Smart Tracking

- Cup capacity: 300ml
- Daily goal: 3000ml (3L)
- Active hours: 8AM - 10PM

</div>

</div>

---

# Why SAM + CV, Not LLM Multimodal?

<div class="flex items-center justify-center h-80">
<span class="text-8xl font-bold">Just For Fun !</span>
</div>



---

# AI Pipeline Deep Dive

<div class="grid grid-cols-3 gap-4">

<div class="text-center">

### Step 1: Detection

<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/sam2fill.jpg" class="h-48 mx-auto rounded" />

SAM2 fill level mask

</div>

<div class="text-center">

### Step 2: Segmentation

<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/regions.jpg" class="h-48 mx-auto rounded" />

Region split analysis

</div>

<div class="text-center">

### Step 3: Level Detection

<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/level.jpg" class="h-48 mx-auto rounded" />

Water level calculation

</div>

</div>

---

# Dual Detection Strategy

<br>

<div class="grid grid-cols-2 gap-8">

<div>

### SAM1 + Classical CV

- Region split analysis
- Brightness change detection

</div>

<div>

### Fine-tuned SAM2

- Custom trained on cup images
- Direct fill ratio calculation

</div>

</div>

<br>

<div class="text-center p-4 bg-green-50 rounded-lg">

**Final level = average of both approaches**

</div>

---

# Dashboard

<div class="grid grid-cols-2 gap-6 mt-2">

<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/Screenshot 2026-04-23 at 3.34.54 PM.png" class="h-96 mx-auto rounded-lg shadow" />

<img src="https://raw.githubusercontent.com/shengyongjiang/picgo_storage/main/Screenshot 2026-04-23 at 3.38.20 PM.png" class="h-96 mx-auto rounded-lg shadow" />

</div>

---

# Tech Stack

<br>

<div class="grid grid-cols-2 gap-8">

<div>

### AI Stack
| Tech | Purpose |
|------|---------|
| YOLO11n | Cup detection |
| SAM (ViT-B) | Segmentation |
| SAM2 (fine-tuned) | Water level |
| OpenCV | Image processing |
| PyTorch | Model inference |

</div>

</div>

---
layout: center
class: text-center
---

# Thank You! 💧

### Stay Hydrated. Stay Healthy.

*Drink, drink, drink, drink...*
<br>
*Drink, drink, drink, drink...*
<br>
*All I have to do is drink~*

<span class="text-xs">(water, of course)</span>

<div class="mt-8 text-sm opacity-60">

Try it yourself: `https://github.com/shengyongjiang/drink_is_all_you_need`

</div>
