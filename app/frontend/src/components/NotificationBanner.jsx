// NotificationBanner — a toast that slides DOWN from the top of the phone whenever the
// status LEVEL changes, then auto-hides after a few seconds.
//
// BUG FIX (see styles.css for the layout half): previously the banner was absolutely
// positioned and transparent, so it OVERLAPPED the "BIKE LANE STATUS" / location text.
// Fixed two ways:
//   1. Solid background (no transparency) via the .notification-banner CSS class.
//   2. It participates in layout and reserves its own height so it PUSHES the phone
//      content down while visible, instead of floating over it. This component renders
//      a spacer wrapper whose height collapses to 0 when hidden, so nothing overlaps.
export default function NotificationBanner({ visible, level, message }) {
  // Always render the wrapper so height changes animate; a `--visible` modifier expands it.
  return (
    <div className={`notification-wrap ${visible ? 'notification-wrap--visible' : ''}`}>
      <div className={`notification-banner notification-banner--${level}`} role="alert">
        <span className={`dot dot--${level}`} />
        <span className="notification-banner__msg">{message}</span>
      </div>
    </div>
  )
}
