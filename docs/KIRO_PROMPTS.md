# Kiro prompts: Comm Ave Bike Lane Monitor

## How to use this file
1. Copy `.kiro/steering/` and `config/camera.yaml` into the root of the Kiro workspace and commit.
2. Each spec below goes into its own Kiro spec session (one teammate each, own git branch).
   Read the generated requirements.md before letting Kiro continue to design and tasks.
3. Order and dependencies:
   - Spec 1 (detection) and Spec 2 (LaneMonitor + counting) can start at the same time.
     Spec 2 is tested on hand-made CSVs, so it doesn't need video yet.
   - Spec 4 (replay demo) can start right away using a fake events.json, then switch to real data.
   - Spec 3 (Telraam, validation, evidence dashboard) starts once some detections exist.
4. Use the vibe prompts at the bottom for small tasks along the way.
5. Keep Kiro in Supervised mode until you trust what it does.

Shared contracts (from tech.md): the detections CSV columns and the events.json format.
If anyone changes either, tell the team and update the other specs.

---

## Spec 1: Detection pipeline (computer vision)

```text
Build src/detect.py and tools/pick_points.py, following the steering files.

Context: we record bike lane video in chunks named commave_YYYY-MM-DD_HHMM.mp4. Each video must
become a CSV of tracked objects so the rest of the team works in pandas without touching video.

Requirements:
1. detect.py takes a video file or a folder (--input) and writes one CSV per video to
   data/detections/<video_name>_detections.csv.
2. Use ultralytics YOLO tracking (persist=True), restricted to the classes in config/camera.yaml.
   Model weights and confidence threshold also come from config/camera.yaml (already exists).
3. CSV columns: video, time (ISO timestamp, America/New_York, from filename start + frame/fps),
   frame, track_id, uid (video name + track_id), cls, conf, cx, cy, w, h (pixels).
4. Options: --stride N (process every Nth frame), --overwrite (otherwise skip videos that already
   have a CSV), --save-annotated (also write an MP4 with boxes and track IDs, with faces and
   plates blurred: top 25% of person boxes, bottom 30% of vehicle boxes).
5. Print progress and a summary: frames processed, detections, unique tracks per class.
6. tools/pick_points.py opens the first frame of a video, lets me click points, and prints them
   as YAML for config/camera.yaml (counting line = 2 points, lane polygon = 4+ points). It saves
   the frame with the shapes drawn on it to data/frames/.
7. detect.py must run in Google Colab (no GUI windows). Add requirements.txt, .gitignore
   (data/raw_video/, demo/media/, *.mp4) and a README section "How to run detection" with Colab
   steps (mount Drive, pip install, run command).

Acceptance: detect.py on a 2 minute test clip produces a CSV with exactly these columns, and the
annotated MP4 plays with faces blurred.

Two cameras: we have two cameras filming the same street at the same time from different
angles. Video names include a camera ID: commave_<camID>_YYYY-MM-DD_HHMM.mp4.
- Each camera has its own config file, config/<camID>.yaml (counting line, lane polygon).
  detect.py and pick_points.py pick the config from the camera ID in the filename.
- Add a "camera" column to the detections CSV, and include the camera ID in uid.
- Never combine counts from different cameras. Each analysis uses one camera.
The videos are already in data/raw_video/. Don't overwrite existing config files.
```

---

## Spec 2: LaneMonitor, blockages and counting (analysis)

```text
Build src/lane_monitor.py, src/blockages.py and src/count.py, following the steering files.
They read the detections CSV (video, time, frame, track_id, uid, cls, conf, cx, cy, w, h) and
never read video.

LaneMonitor (src/lane_monitor.py), the most important piece:
1. A class LaneMonitor(config) with a method update(time, detections_for_this_frame) that returns
   a list of new events (often empty). It must work one frame at a time, so the same class can
   later run on a live feed. A helper run_on_dataframe(df) feeds a whole CSV through it.
2. A vehicle (class in blockage.vehicle_classes) is "in the lane" when the bottom-center of its
   box (cx, cy + h/2) is inside lane_polygon, and "stopped" when its speed, averaged over the
   last 2 seconds, is below stopped_max_speed_px_s.
3. Lane status is a state machine: clear -> possible (a vehicle stopped in the lane for
   possible_after_s) -> blocked (blocked_after_s) -> clear (no stopped vehicle for clear_after_s).
   A vehicle lost by the tracker and re-found within merge_gap_s and merge_distance_px is the
   same vehicle (the tracker sometimes gives a parked truck a new ID when someone walks past).
4. Events are dicts: {"event_id", "type": "possible" | "blocked" | "cleared", "time" (ISO),
   "vehicle" (car/truck/bus), "x", "y" (pixel position), "stopped_for_s"}; "cleared" also has
   "duration_s". Every blockage has a "blocked" and a matching "cleared" event.

Batch outputs (src/blockages.py):
5. Run LaneMonitor over one CSV or a folder. Write outputs/blockages.csv (event_id, start, end,
   duration_seconds, vehicle_type, x, y, reached_blocked true/false) and outputs/events.jsonl.
   Print a summary: blockages, total minutes blocked, longest blockage, blockages per hour.

Counting (src/count.py):
6. A track is counted once, the first time its center crosses counting_line. Record time and
   direction using side_a_name / side_b_name from config.
7. Classify crossing tracks as: bike; pedestrian (person that does NOT overlap a bicycle box in
   more than 50% of its frames, since those are riders); skateboard; possible_scooter (person, not
   a rider, faster than scooter.min_speed_px_s); vehicle.
8. Write outputs/crossings.csv (one row per counted object: time, uid, mode, direction) and
   outputs/hourly_counts.csv (hour x mode, plus direction breakdown).

Tests (pytest, small hand-made CSVs in tests/fixtures/): a bike crosses once; a track wobbling on
the line counts once; a person riding a bike is not a pedestrian; a car stopped 45 s in the lane
gives possible -> blocked -> cleared; a car stopped 8 s gives no events; a car driving through
the lane gives no events; one parked truck split into two track IDs gives ONE blockage.
```

---

## Spec 3: Telraam, validation and evidence dashboard (data and presentation)

```text
Build src/telraam.py, src/validate.py and src/dashboard.py, following the steering files.

Telraam (telraam.py):
1. Download hourly counts for one road segment from the Telraam API
   (POST https://telraam-api.net/v1/reports/traffic, header X-Api-Key). Segment ID and date range
   from CLI args; API key from the environment variable TELRAAM_API_KEY, never in code.
   Check the current API docs for the exact request and response fields.
2. Save to data/telraam/telraam_<segment>_<start>_<end>.csv with times in America/New_York.
   Keep the uptime / camera-activity column.

Validation (validate.py):
3. Join outputs/hourly_counts.csv with Telraam by hour. Only compare hours our camera recorded
   in full and where Telraam uptime >= validation.min_telraam_uptime.
4. For bikes and pedestrians report: ours, Telraam, difference, percent difference, and overall
   mean absolute percent error. Save outputs/validation.csv.
5. Also accept a manual count CSV (clip_start, clip_end, mode, count) and compare it with our
   crossings in the same window. Print one sentence we can put on a slide, e.g.
   "Bike counts were within X% of a hand count and Y% of the Telraam sensor."

Evidence dashboard (dashboard.py):
6. One self-contained HTML file outputs/dashboard.html (Plotly, with plotly.js embedded so it works
   offline):
   - headline numbers: blockages, total minutes blocked, longest blockage, bikes counted
   - blockage timeline: one bar per event from start to end, colored by vehicle type
   - blockages by hour of day (bar)
   - hourly counts by mode (line), and ours vs Telraam (grouped bar) with the error in the subtitle
   - map of the camera location (config) and, if data/public/vision_zero.csv exists, bike crashes
     within 300 m
7. Chart titles state the finding (e.g. "Most blockages happen 11am-1pm"), axes are labeled, and
   each mode keeps the same color in every chart.

Acceptance: running the three scripts on sample data produces validation.csv and a dashboard.html
that opens in a browser with no internet.
```

---

## Spec 4: Replay demo (main deliverable)

```text
Build the replay demo: src/build_demo.py and demo/index.html, following the steering files.

Goal: on stage, a recorded clip of the bike lane plays in a browser. Next to it, a phone-sized
"rider app" shows the lane status and alerts as they happen in the video. Everything shown in the
app comes from our pipeline (LaneMonitor in src/lane_monitor.py), precomputed and played back in
sync with the video. It must run offline.

build_demo.py:
1. Inputs: --video (a raw chunk), --start and --end (seconds within that chunk, to cut a 2 to 4
   minute clip), --detections (that chunk's CSV). If no CSV is given, run detection on the clip.
2. Run the clip's detections through LaneMonitor (same config as the batch analysis) and the
   counting line logic from count.py.
3. Write demo/media/demo_clip.mp4: the clip with faces/plates blurred, the lane polygon drawn and
   filled green / amber / red according to lane status at that moment, boxes around vehicles that
   are stopped in the lane, and a small timestamp. Encode H.264 so browsers can play it.
4. Write demo/data/events.json in this format (times are seconds from the start of the clip):
   {
     "location": {"name": str, "lat": float, "lon": float},
     "clip_start_time": ISO timestamp of the first frame,
     "video": "media/demo_clip.mp4",
     "events": [{"t": float, "type": "possible"|"blocked"|"cleared", "vehicle": str,
                 "stopped_for_s": float, "duration_s": float (cleared only), "event_id": int}],
     "crossings": [{"t": float, "mode": "bike"|"pedestrian"|"skateboard"|"possible_scooter"}],
     "summary": {"blockages": int, "minutes_blocked": float, "bikes": int}
   }
5. Also write demo/data/sample_events.json with made-up events so the front end can be built
   and tested before real data exists.

demo/index.html (single file, vanilla JS, no external libraries):
6. Layout: video on the left (about 65% width), a phone mockup on the right. Below the video:
   play/pause, restart, and speed buttons (1x, 2x, 4x) plus a scrubbable timeline with colored
   markers at each event.
7. The app reads events.json and, on every video timeupdate, derives the status from the latest
   event at or before video.currentTime. It must stay correct when the user seeks or changes speed.
8. Phone screen shows: location name; a large status card (green "Lane clear", amber "Vehicle
   stopped in lane", red "Bike lane blocked - <vehicle>, <mm:ss>" with a live-counting duration);
   a slide-down notification banner with a short sound on each "blocked" and "cleared" event
   (sound can be muted); a recent alerts list; "bikes in the last 10 minutes" from crossings.
9. Load events from data/events.json, falling back to data/sample_events.json, with a visible
   note if the sample file is being used. Works when served with `python -m http.server` from the
   demo/ folder. Large, readable text for a projector.
10. README section "Running the demo": exact commands, and a pre-demo checklist.

Optional stretch (only if everything above works): a --live flag on a separate script
src/live_demo.py that reads a camera stream with OpenCV, runs YOLO + LaneMonitor in real time,
and serves the same page with events pushed over Server-Sent Events. The page must work
unchanged in replay mode.

Acceptance: with sample_events.json the page shows clear -> possible -> blocked -> cleared at the
right moments, including after seeking backwards; with a real clip the red status appears while
the vehicle is visibly stopped in the lane.
```

---

## Vibe prompts (quick tasks)

**First prototype before any spec**
```text
I have a video at data/raw_video/test.mp4. Write a quick script that runs YOLO tracking on it,
counts unique bicycles and people, and prints the counts. Under 40 lines. Run it.
```

**Check the camera angle**
```text
Run detect.py on data/raw_video/test.mp4 with --save-annotated, then tell me how many tracks per
class it found and whether bikes seem to get new track IDs partway across the frame.
```

**Find a good demo clip**
```text
Using outputs/blockages.csv, list the 5 blockages that would make the best demo clip: reached
"blocked", lasted 45 s to 3 min, and had bikes crossing during the blockage (from crossings.csv).
For each, give the video file and the --start/--end values for build_demo.py with 20 s of padding.
```

**Debug a missed blockage**
```text
In commave_2026-10-24_1200 there's a delivery van stopped in the lane from about 4:10 to 5:30,
but LaneMonitor never reaches "blocked". Look at that van's tracks in the detections CSV: is its
bottom-center inside the lane polygon, is its speed under the threshold, did its track ID change?
Explain before changing code.
```

**Debug double counts**
```text
count.py says 212 bikes in commave_2026-10-24_0800 but we hand-counted about 150. Find out why
using crossings.csv and the detections CSV. Explain what you find before changing code.
```

**Tune thresholds**
```text
Plot a histogram of how long vehicles stay stopped in the lane polygon across all detection CSVs,
so we can choose possible_after_s and blocked_after_s. Save it to outputs/.
```

**Polish the demo**
```text
Make demo/index.html look polished on a 1080p projector: larger fonts, a dark theme, smooth
banner animation, and the phone mockup with rounded corners and a notch. Don't change the
events.json logic.
```

**Explain code to a teammate**
```text
Explain how LaneMonitor decides a lane is blocked, in plain language for someone new to Python.
Don't change any code.
```

**Slides chart**
```text
Make a PNG for our slides from outputs/blockages.csv: blockages by hour of day, title stating the
busiest hour, fonts readable on a projector. Save to outputs/.
```
