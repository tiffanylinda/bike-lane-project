import { useEffect, useRef, useState } from 'react'
import { CONFIG } from '../config.js'
import { mmss } from '../lib/format.js'
import Timeline from './Timeline.jsx'

// VideoPanel — the left column: the recorded clip plus the replay controls and timeline.
//
// The replay CLOCK (from useReplayClock) is the source of truth. The <video> is slaved to
// it: we set video.currentTime on seek, mirror play/pause, and apply playbackRate for
// speed. WHY: this keeps the phone UI and the video in sync, and lets the whole demo run
// even when the video is missing (GitHub Pages) by driving everything off the clock and
// CONFIG.videoDurationSeconds.
export default function VideoPanel({
  currentTime,
  playing,
  speed,
  controls,
  duration,
  events,
  onVideoDuration,
}) {
  const videoRef = useRef(null)
  const [muted, setMuted] = useState(true) // muted by default; Sound toggle controls it
  const [videoFailed, setVideoFailed] = useState(false)

  // Keep video.currentTime aligned with the clock. We only correct when they drift apart
  // by more than ~0.3s to avoid fighting the browser's own playback progression.
  useEffect(() => {
    const video = videoRef.current
    if (!video || videoFailed) return
    if (Math.abs(video.currentTime - currentTime) > 0.3) {
      video.currentTime = currentTime
    }
  }, [currentTime, videoFailed])

  // Mirror play/pause from the clock to the video element.
  useEffect(() => {
    const video = videoRef.current
    if (!video || videoFailed) return
    if (playing) {
      video.play().catch(() => {
        // Autoplay can be blocked; the clock keeps running regardless.
      })
    } else {
      video.pause()
    }
  }, [playing, videoFailed])

  // Apply speed to the video's native playbackRate.
  useEffect(() => {
    const video = videoRef.current
    if (!video || videoFailed) return
    video.playbackRate = speed
  }, [speed, videoFailed])

  // Apply mute state.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.muted = muted
  }, [muted])

  const speeds = [1, 2, 4]

  return (
    <section className="video-panel">
      <div className="video-stage">
        {videoFailed ? (
          // Placeholder shown when the clip is absent/unplayable (e.g. on Pages). The
          // timeline and clock below still work off CONFIG.videoDurationSeconds.
          <div className="video-placeholder">
            <div className="video-placeholder__title">No video loaded</div>
            <div className="video-placeholder__hint">
              The replay clip is local-only. Status and alerts still play back below.
            </div>
          </div>
        ) : (
          <video
            ref={videoRef}
            className="video-el"
            src={CONFIG.videoSrc}
            muted={muted}
            playsInline
            preload="metadata"
            // No native controls: the replay controls below drive everything.
            onLoadedMetadata={(e) => {
              const d = e.currentTarget.duration
              if (Number.isFinite(d) && d > 0) onVideoDuration(d)
            }}
            onError={() => setVideoFailed(true)}
          />
        )}
      </div>

      <div className="controls">
        <button className="btn" onClick={controls.toggle}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button className="btn" onClick={controls.restart}>
          Restart
        </button>

        <div className="controls__speed">
          {speeds.map((s) => (
            <button
              key={s}
              className={`btn btn--speed ${speed === s ? 'btn--active' : ''}`}
              onClick={() => controls.setSpeed(s)}
            >
              {s}x
            </button>
          ))}
        </div>

        <button className="btn" onClick={() => setMuted((m) => !m)}>
          Sound {muted ? 'off' : 'on'}
        </button>

        <span className="controls__time">
          {mmss(currentTime)} / {mmss(duration)}
        </span>
      </div>

      <Timeline
        currentTime={currentTime}
        duration={duration}
        events={events}
        onSeek={controls.seek}
      />
    </section>
  )
}
