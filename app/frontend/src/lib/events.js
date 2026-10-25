// events.js — load the replay events, with a sample-data fallback.
//
// WHY the fallback: on GitHub Pages (or any static host) we want the demo to work even if
// the real data/events.json is missing. If the primary fetch fails, we load
// data/sample_events.json instead and flag usedSample so the UI can warn the viewer that
// they are looking at sample data rather than our real blockage events.

// Fetch a JSON file relative to the app base. We use a relative path (no leading slash)
// so it resolves under the GitHub Pages sub-path and locally alike.
async function fetchJson(path) {
  const res = await fetch(path)
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status}`)
  }
  return res.json()
}

/**
 * loadEvents — try the real events first, fall back to the sample set.
 * @returns {Promise<{ events: Array, usedSample: boolean }>}
 */
export async function loadEvents() {
  try {
    const events = await fetchJson('data/events.json')
    return { events, usedSample: false }
  } catch (primaryError) {
    // Primary file missing or unreadable — try the sample set so the demo still runs.
    console.warn('events.json unavailable, falling back to sample_events.json', primaryError)
    try {
      const events = await fetchJson('data/sample_events.json')
      return { events, usedSample: true }
    } catch (sampleError) {
      // Both failed. Return an empty set so the app renders a working (if empty) demo
      // instead of crashing.
      console.error('sample_events.json also unavailable', sampleError)
      return { events: [], usedSample: true }
    }
  }
}
