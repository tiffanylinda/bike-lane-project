import { CONFIG } from '../config.js'
import StatusCard from './StatusCard.jsx'
import RerouteCard from './RerouteCard.jsx'
import RecentAlerts from './RecentAlerts.jsx'
import BikeCounter from './BikeCounter.jsx'
import CrashHistory from './CrashHistory.jsx'
import NotificationBanner from './NotificationBanner.jsx'

// PhoneMock — the phone-sized card (middle column) holding the full rider UI. App passes
// the current status, the derived alerts, and the notification state down; PhoneMock just
// lays them out inside a phone frame.
export default function PhoneMock({ status, alerts, currentTime, notification }) {
  return (
    <div className="phone">
      <div className="phone__frame">
        {/* Banner lives at the TOP and, when visible, pushes the content below it down
            (see NotificationBanner + styles.css) so it never overlaps the heading. */}
        <NotificationBanner
          visible={notification.visible}
          level={notification.level}
          message={notification.message}
        />

        <div className="phone__content">
          <div className="phone__heading">BIKE LANE STATUS</div>
          <div className="phone__location">{CONFIG.location}</div>

          <StatusCard status={status} />

          {/* Reroute only appears while blocked — exactly when a rider needs it. */}
          {status.level === 'blocked' && <RerouteCard />}

          <RecentAlerts alerts={alerts} />
          <BikeCounter currentTime={currentTime} />
          <CrashHistory />
        </div>
      </div>
    </div>
  )
}
