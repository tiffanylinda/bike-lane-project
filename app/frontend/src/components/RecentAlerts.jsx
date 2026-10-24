import { mmss } from '../lib/format.js'

// RecentAlerts — history of alert points that have happened up to the current time.
// The list is derived by alertsUpTo() (pure), so it grows while playing and shrinks when
// you seek backward. Newest alert is shown first.
export default function RecentAlerts({ alerts }) {
  if (alerts.length === 0) {
    return (
      <div className="recent-alerts">
        <div className="recent-alerts__heading">Recent alerts</div>
        <div className="recent-alerts__empty">No alerts yet.</div>
      </div>
    )
  }

  return (
    <div className="recent-alerts">
      <div className="recent-alerts__heading">Recent alerts</div>
      <ul className="recent-alerts__list">
        {[...alerts].reverse().map((a, i) => {
          const message =
            a.level === 'blocked'
              ? `${a.vehicle} blocking the lane`
              : `${a.vehicle} stopped in the lane`
          return (
            <li key={i} className="recent-alerts__item">
              <span className={`dot dot--${a.level}`} />
              <span className="recent-alerts__msg">{message}</span>
              <span className="recent-alerts__time">{mmss(a.time)}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
