// config.js — SINGLE SOURCE OF TRUTH for every tunable number, label and path.
// Change behaviour here, not inside components, so the demo stays easy to retune.

export const CONFIG = {
  // Backend API base URL. Reads from the Vite env at build time with a local default.
  // Set VITE_API_BASE="" (empty) to force static-JSON mode — e.g. for GitHub Pages,
  // where there is no backend to reach and we want to serve the bundled JSON instead.
  apiBase: import.meta.env.VITE_API_BASE || 'http://localhost:8000',

  // Relative path (no leading slash) so it resolves the same way locally and on
  // GitHub Pages project sites. The clip may be absent on Pages; the app still runs.
  videoSrc: 'media/demo_clip.mov',

  // Fallback timeline length (seconds) used when the video is missing or its metadata
  // never loads. Keeps the clock, timeline and status logic working with no video.
  videoDurationSeconds: 120,

  // Product-steering timing, confirmed by the team. These drive status.js.
  thresholds: {
    possibleAfterSeconds: 10, // amber: a vehicle has been stopped at least 10s
    blockedAfterSeconds: 30,  // red: stopped at least 30s
    clearDebounceSeconds: 5,  // stay "clear" for a few seconds after an event ends (smoothing)
  },

  // Where this bike lane is, shown on the rider phone.
  location: 'Comm Ave bike lane, Essex St to Kenmore Sq',

  // A STATIC, illustrative reroute shown only while the lane is blocked. This is a demo
  // suggestion, not a real routing result.
  reroute: {
    name: 'Detour via Bay State Rd',
    extraMinutes: 3,
    description: 'Illustrative suggestion for this demo scenario.',
  },

  // Label for the bike counter widget on the phone.
  bikesWindowLabel: 'bikes in the last 10 minutes',
}
