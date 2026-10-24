// CrashHistory — a few illustrative crash records for context on the phone. Static demo
// data (not live); real Vision Zero crash data would feed this in the full product.
const ROWS = [
  { date: '2023-08-14', type: 'Cyclist vs. right-turning car', severity: 'Injury' },
  { date: '2023-05-02', type: 'Dooring near parked cars', severity: 'Minor' },
  { date: '2022-11-19', type: 'Cyclist vs. double-parked truck', severity: 'Injury' },
]

export default function CrashHistory() {
  return (
    <div className="crash-history">
      <div className="crash-history__heading">Bike crash history (Comm Ave)</div>
      <ul className="crash-history__list">
        {ROWS.map((r, i) => (
          <li key={i} className="crash-history__row">
            <span className="crash-history__date">{r.date}</span>
            <span className="crash-history__type">{r.type}</span>
            <span className="crash-history__sev">{r.severity}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
