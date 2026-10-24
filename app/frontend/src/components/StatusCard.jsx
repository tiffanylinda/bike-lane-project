import { blockedLabel } from '../lib/format.js'

// StatusCard — the big colored status block on the phone. Its text/color come straight
// from the pure status level, so it always matches statusAt().
export default function StatusCard({ status }) {
  const { level, vehicle, blockedForSeconds } = status

  // Pick the headline + sub-line for each level.
  let title
  let subtitle
  if (level === 'blocked') {
    title = 'Bike lane blocked'
    subtitle = `${vehicle} in the lane`
  } else if (level === 'possible') {
    title = 'Vehicle stopped in lane'
    subtitle = `Watching a ${vehicle}...`
  } else {
    title = 'Lane clear'
    subtitle = 'The bike lane is open.'
  }

  return (
    <div className={`status-card status-card--${level}`}>
      <div className="status-card__title">{title}</div>
      <div className="status-card__subtitle">{subtitle}</div>

      {/* Live "blocked for" counter, only while blocked. Reads e.g. "1 min" or "0:42". */}
      {level === 'blocked' && (
        <div className="status-card__timer">Blocked for {blockedLabel(blockedForSeconds)}</div>
      )}
    </div>
  )
}
