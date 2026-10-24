import { CONFIG } from '../config.js'

// Timeline — a seekable bar with colored markers for each event.
//
// Each event is drawn as two stacked segments positioned by its start/end over the total
// duration: an amber band from (start + possibleAfter) and a red band from
// (start + blockedAfter), matching when the status actually turns amber/red. Clicking or
// dragging anywhere on the bar seeks BOTH the replay clock and the video (App wires seek).
export default function Timeline({ currentTime, duration, events, onSeek }) {
  const { possibleAfterSeconds, blockedAfterSeconds } = CONFIG.thresholds

  // Convert a click/touch x-position into a time and seek there.
  const handlePointer = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const fraction = Math.max(0, Math.min(1, x / rect.width))
    onSeek(fraction * duration)
  }

  const pct = (seconds) => `${(seconds / duration) * 100}%`

  return (
    <div
      className="timeline"
      onClick={handlePointer}
      role="slider"
      aria-label="Seek replay"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(currentTime)}
      tabIndex={0}
    >
      {events.map((ev, i) => {
        const amberStart = ev.start + possibleAfterSeconds
        const redStart = ev.start + blockedAfterSeconds
        return (
          <div key={i}>
            {/* Amber band: from when it becomes "possible" until it becomes "blocked" (or ends). */}
            {amberStart < ev.end && (
              <div
                className="timeline__marker timeline__marker--possible"
                style={{
                  left: pct(amberStart),
                  width: pct(Math.min(redStart, ev.end) - amberStart),
                }}
                title={`${ev.vehicle}: watching`}
              />
            )}
            {/* Red band: from when it becomes "blocked" until it ends. */}
            {redStart < ev.end && (
              <div
                className="timeline__marker timeline__marker--blocked"
                style={{ left: pct(redStart), width: pct(ev.end - redStart) }}
                title={`${ev.vehicle}: blocked`}
              />
            )}
          </div>
        )
      })}

      {/* Progress fill + playhead at the current time. */}
      <div className="timeline__progress" style={{ width: pct(currentTime) }} />
      <div className="timeline__playhead" style={{ left: pct(currentTime) }} />
    </div>
  )
}
