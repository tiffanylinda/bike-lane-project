import { useCallback, useEffect, useRef, useState } from 'react'

// useReplayClock — the heartbeat of the replay.
//
// WHY a dedicated clock (instead of just reading video.currentTime): the demo must work
// WITHOUT the video on GitHub Pages. So the clock is the source of truth for "where are we
// in the replay". The video, when present, is slaved to this clock (App sets
// video.currentTime / playbackRate). This keeps the phone UI and the video in sync and
// lets everything keep running when the video is absent.
//
// It advances currentTime using REAL elapsed wall-clock time multiplied by speed, measured
// with requestAnimationFrame. Using elapsed-time deltas (not a fixed +1/frame) means speed
// changes and frame-rate hiccups never cause drift.

export function useReplayClock(durationSeconds) {
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

  // Refs hold values the animation loop reads without re-subscribing every render.
  const rafRef = useRef(null)
  const lastTsRef = useRef(null)
  const speedRef = useRef(speed)
  const durationRef = useRef(durationSeconds)

  useEffect(() => {
    speedRef.current = speed
  }, [speed])

  useEffect(() => {
    durationRef.current = durationSeconds
  }, [durationSeconds])

  // The animation loop. Runs only while playing.
  useEffect(() => {
    if (!playing) {
      lastTsRef.current = null
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      return
    }

    const tick = (ts) => {
      if (lastTsRef.current === null) {
        lastTsRef.current = ts
      }
      const elapsedSeconds = (ts - lastTsRef.current) / 1000
      lastTsRef.current = ts

      setCurrentTime((prev) => {
        const next = prev + elapsedSeconds * speedRef.current
        // Stop cleanly at the end of the replay.
        if (next >= durationRef.current) {
          setPlaying(false)
          return durationRef.current
        }
        return next
      })

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [playing])

  // --- Controls: stable callbacks the UI buttons call. ---
  const play = useCallback(() => setPlaying(true), [])
  const pause = useCallback(() => setPlaying(false), [])
  const toggle = useCallback(() => setPlaying((p) => !p), [])

  const seek = useCallback((t) => {
    // Clamp into [0, duration] so dragging the timeline can't go out of range.
    const clamped = Math.max(0, Math.min(t, durationRef.current))
    lastTsRef.current = null // reset the delta base so the next tick doesn't jump
    setCurrentTime(clamped)
  }, [])

  const restart = useCallback(() => {
    lastTsRef.current = null
    setCurrentTime(0)
    setPlaying(true)
  }, [])

  const changeSpeed = useCallback((s) => setSpeed(s), [])

  return {
    currentTime,
    playing,
    speed,
    controls: { play, pause, toggle, seek, restart, setSpeed: changeSpeed },
  }
}
