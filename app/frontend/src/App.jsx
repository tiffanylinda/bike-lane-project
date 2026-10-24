import { useEffect, useMemo, useRef, useState } from 'react'
import { CONFIG } from './config.js'
import { loadEvents } from './lib/events.js'
import { statusAt, alertsUpTo } from './lib/status.js'
import { useReplayClock } from './hooks/useReplayClock.js'
import Header from './components/Header.jsx'
import VideoPanel from './components/VideoPanel.jsx'
import PhoneMock from './components/PhoneMock.jsx'
import LockScreenWidget from './components/LockScreenWidget.jsx'
import SampleDataWarning from './components/SampleDataWarning.jsx'

// How long the slide-down notification stays visible after a status change (ms).
const NOTIFICATION_MS = 4000

export default function App() {
  // --- Load events once on mount, with sample-data fallback. ---
  const [events, setEvents] = useState([])
  const [usedSample, setUsedSample] = useState(false)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    loadEvents().then(({ events, usedSample }) => {
      if (cancelled) return
      setEvents(events)
      setUsedSample(usedSample)
      setLoaded(true)
    })
    return () => {
      cancelled = true
    }
  }, [])

  // --- Duration: prefer the real video's metadata, fall back to the configured value. ---
  const [duration, setDuration] = useState(CONFIG.videoDurationSeconds)

  // --- The replay clock drives everything. ---
  const { currentTime, playing, speed, controls } = useReplayClock(duration)

  // --- Derive status + alerts purely from the current time. No drift on seek/speed/restart. ---
  const status = useMemo(
    () => statusAt(currentTime, events, CONFIG.thresholds),
    [currentTime, events],
  )
  const alerts = useMemo(
    () => alertsUpTo(currentTime, events, CONFIG.thresholds),
    [currentTime, events],
  )

  // --- Notification banner: fire when the status LEVEL changes. ---
  const [notification, setNotification] = useState({ visible: false, level: 'clear', message: '' })
  const prevLevelRef = useRef(status.level)
  const hideTimerRef = useRef(null)

  useEffect(() => {
    const prev = prevLevelRef.current
    if (status.level !== prev) {
      prevLevelRef.current = status.level

      // Compose a short message for the new level.
      let message
      if (status.level === 'blocked') message = `Bike lane blocked — ${status.vehicle} in the lane`
      else if (status.level === 'possible') message = `Vehicle stopped — watching a ${status.vehicle}`
      else message = 'Lane clear — the bike lane is open'

      setNotification({ visible: true, level: status.level, message })

      // Auto-hide after a few seconds. Reset any pending timer first.
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
      hideTimerRef.current = setTimeout(() => {
        setNotification((n) => ({ ...n, visible: false }))
      }, NOTIFICATION_MS)
    }
  }, [status.level, status.vehicle])

  // Clean up the timer on unmount.
  useEffect(() => () => hideTimerRef.current && clearTimeout(hideTimerRef.current), [])

  // When the video reports its real duration, adopt it as the timeline length.
  const handleVideoDuration = (d) => setDuration(d)

  return (
    <div className="app">
      <Header />
      {usedSample && <SampleDataWarning />}

      {loaded ? (
        <main className="layout">
          <VideoPanel
            currentTime={currentTime}
            playing={playing}
            speed={speed}
            controls={controls}
            duration={duration}
            events={events}
            onVideoDuration={handleVideoDuration}
          />
          <PhoneMock
            status={status}
            alerts={alerts}
            currentTime={currentTime}
            notification={notification}
          />
          <LockScreenWidget status={status} />
        </main>
      ) : (
        <div className="loading">Loading replay…</div>
      )}
    </div>
  )
}
