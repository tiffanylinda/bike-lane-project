# Product: Comm Ave Bike Lane Monitor

Hackathon project (Spark! / Hackachusetts). We analyze bike lane usage and blockages on
Commonwealth Ave between Essex St and Kenmore Square from our own camera footage, and we
demo a rider-facing blockage alert.

Goals, in priority order:
1. Detect bike lane blockages: a car, truck or bus stopped in the lane. Lane status goes
   clear -> possible (vehicle stopped 10 s) -> blocked (stopped 30 s) -> clear.
2. Count bikes and pedestrians per hour from video (skateboards and scooters if possible).
3. Validate counts against a Telraam sensor and a manual count.
4. MAIN DELIVERABLE: a replay demo. A recorded clip plays in a browser next to a
   phone-sized "rider app" that shows lane status and alerts produced by our pipeline.
5. Supporting evidence: blockage timeline, incident map (with Vision Zero crashes),
   hourly activity charts, a one-page findings summary with a recommendation.
6. Stretch, only if 1 to 4 already work: run the same pipeline on a live camera feed.

Users are a small student team with intermediate Python. Prefer simple, readable code over
clever code. Everything must run on a laptop or free Google Colab. The demo must work
offline (no internet at the venue is assumed).
