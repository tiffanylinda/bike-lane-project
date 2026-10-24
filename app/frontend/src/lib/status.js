// status.js — PURE status logic for the rider phone.
//
// WHY pure: statusAt() depends ONLY on its arguments (t, events, thresholds). It reads no
// clock and holds no state. That means seeking, restarting, or changing playback speed all
// recompute the exact same answer for a given time with zero drift — the UI just asks
// "what is the status at second t?" and trusts the result.

// Find the event that is active at time t, if any.
// An event is "active" when event.start <= t <= event.end.
// If events overlapped we'd pick the first match; the demo data does not overlap.
function activeEventAt(t, events) {
  for (const event of events) {
    if (t >= event.start && t <= event.end) {
      return event
    }
  }
  return null
}

// Find the most recent event that ENDED at or before t (used for the clear-debounce
// smoothing window, so the phone does not flicker back to "clear" the instant an event ends).
function lastEndedEventAt(t, events) {
  let latest = null
  for (const event of events) {
    if (event.end <= t) {
      if (latest === null || event.end > latest.end) {
        latest = event
      }
    }
  }
  return latest
}

/**
 * statusAt — compute the lane status at a point in time.
 *
 * @param {number} t         seconds into the recording
 * @param {Array}  events    [{ start, end, vehicle, camera, note }, ...] (seconds)
 * @param {object} thresholds { possibleAfterSeconds, blockedAfterSeconds, clearDebounceSeconds }
 * @returns {{ level:'clear'|'possible'|'blocked', vehicle:?string, event:?object, blockedForSeconds:number }}
 */
export function statusAt(t, events, thresholds) {
  const { possibleAfterSeconds, blockedAfterSeconds, clearDebounceSeconds } = thresholds

  const event = activeEventAt(t, events)

  // No event active right now. We are "clear". (The debounce window below is handled
  // implicitly: once an event ends there is no active event, so we are clear — the
  // debounce value is available for callers that want to suppress rapid re-alerts.)
  if (!event) {
    // Keep a reference to the just-ended event during the smoothing window so callers
    // could, if desired, know what we were just watching. Status itself stays "clear".
    const recentlyEnded = lastEndedEventAt(t, events)
    const withinDebounce =
      recentlyEnded !== null && t - recentlyEnded.end <= clearDebounceSeconds
    return {
      level: 'clear',
      vehicle: withinDebounce ? recentlyEnded.vehicle : null,
      event: null,
      blockedForSeconds: 0,
    }
  }

  // An event is active. How long has the vehicle been stopped? Measured from event.start.
  const stoppedForSeconds = t - event.start

  if (stoppedForSeconds >= blockedAfterSeconds) {
    // Red: blocked. blockedForSeconds is a LIVE counter from the start of the event.
    return {
      level: 'blocked',
      vehicle: event.vehicle,
      event,
      blockedForSeconds: stoppedForSeconds,
    }
  }

  if (stoppedForSeconds >= possibleAfterSeconds) {
    // Amber: possible. Stopped long enough to watch, not long enough to call blocked.
    return {
      level: 'possible',
      vehicle: event.vehicle,
      event,
      blockedForSeconds: 0,
    }
  }

  // Event active but stopped for < possibleAfterSeconds: still treat the lane as clear.
  // (A vehicle that only just stopped is not yet an alert.)
  return {
    level: 'clear',
    vehicle: event.vehicle,
    event,
    blockedForSeconds: 0,
  }
}

/**
 * alertsUpTo — derive the list of alert points that have occurred by time t.
 *
 * WHY: RecentAlerts shows a history. An event generates an alert when it first reaches
 * 'possible' (start + possibleAfterSeconds) and again when it reaches 'blocked'
 * (start + blockedAfterSeconds). We only include alert points whose time <= t, so the
 * list grows naturally as the replay plays and shrinks correctly when you seek backward.
 *
 * @returns {Array<{ level:'possible'|'blocked', vehicle:string, time:number, note:string }>}
 *          sorted by time ascending.
 */
export function alertsUpTo(t, events, thresholds) {
  const { possibleAfterSeconds, blockedAfterSeconds } = thresholds
  const alerts = []

  for (const event of events) {
    const possibleAt = event.start + possibleAfterSeconds
    const blockedAt = event.start + blockedAfterSeconds

    // Only fire the "possible" alert if the event actually lasts that long AND we've reached it.
    if (possibleAt <= event.end && t >= possibleAt) {
      alerts.push({
        level: 'possible',
        vehicle: event.vehicle,
        time: possibleAt,
        note: event.note,
      })
    }
    if (blockedAt <= event.end && t >= blockedAt) {
      alerts.push({
        level: 'blocked',
        vehicle: event.vehicle,
        time: blockedAt,
        note: event.note,
      })
    }
  }

  alerts.sort((a, b) => a.time - b.time)
  return alerts
}
