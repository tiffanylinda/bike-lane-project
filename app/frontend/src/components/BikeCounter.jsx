import { CONFIG } from '../config.js'

// BikeCounter — a small "N bikes in the last 10 minutes" widget. For the replay demo we
// show a reasonable derived number (varies a little over the replay so it feels live).
// WHY derived rather than hard-coded: it reads more convincingly while the clip plays.
export default function BikeCounter({ currentTime }) {
  // Start from a baseline and nudge it up slowly as the replay progresses. Deterministic
  // (a pure function of currentTime) so seeking shows a consistent number.
  const baseline = 12
  const extra = Math.floor(currentTime / 15) // +1 roughly every 15s of replay
  const count = baseline + extra

  return (
    <div className="bike-counter">
      <span className="bike-counter__num">{count}</span>
      <span className="bike-counter__label">{CONFIG.bikesWindowLabel}</span>
    </div>
  )
}
