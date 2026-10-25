import { CONFIG } from '../config.js'
import { blockedLabel } from '../lib/format.js'

// LockScreenWidget — far-right column. A preview of an iOS-style lock-screen widget that
// mirrors the current status at a glance (color + short text), so the audience sees how a
// rider would get the alert without opening the app.
export default function LockScreenWidget({ status }) {
  const { level, vehicle, blockedForSeconds } = status

  let short
  if (level === 'blocked') short = `${vehicle} blocking lane`
  else if (level === 'possible') short = `${vehicle} stopped`
  else short = 'Lane clear'

  return (
    <aside className="lockscreen">
      <div className="lockscreen__label">Lock-screen widget preview</div>

      <div className="lockscreen__phone">
        <div className="lockscreen__clock">9:41</div>
        <div className="lockscreen__date">Comm Ave</div>

        <div className={`lockscreen__widget lockscreen__widget--${level}`}>
          <div className="lockscreen__widget-title">Bike Lane</div>
          <div className="lockscreen__widget-status">{short}</div>
          {level === 'blocked' && (
            <div className="lockscreen__widget-sub">{blockedLabel(blockedForSeconds)}</div>
          )}
          {level === 'clear' && (
            <div className="lockscreen__widget-sub">{CONFIG.location}</div>
          )}
        </div>
      </div>
    </aside>
  )
}
