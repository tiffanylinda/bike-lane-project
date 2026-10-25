// format.js — tiny shared formatting helpers used across components.

// Format seconds as m:ss (e.g. 65 -> "1:05"). Used for clock readouts and alert times.
export function mmss(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds))
  const minutes = Math.floor(s / 60)
  const seconds = s % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

// Human "blocked for" label. Under a minute shows m:ss; a minute or more rounds to
// "<n> min" so the phone reads "truck, 1 min" as described in the demo.
export function blockedLabel(seconds) {
  if (seconds >= 60) {
    const mins = Math.floor(seconds / 60)
    return `${mins} min`
  }
  return mmss(seconds)
}
