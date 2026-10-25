import { CONFIG } from '../config.js'

// RerouteCard — a static, illustrative detour suggestion. App renders this only while the
// lane is blocked, so the rider sees an alternative exactly when it matters.
export default function RerouteCard() {
  const { name, extraMinutes, description } = CONFIG.reroute
  return (
    <div className="reroute-card">
      <div className="reroute-card__name">{name}</div>
      <div className="reroute-card__extra">Est. +{extraMinutes} min</div>
      <div className="reroute-card__desc">{description}</div>
    </div>
  )
}
