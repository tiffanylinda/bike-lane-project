// SampleDataWarning — shown only when the real events.json was missing and we fell back
// to sample_events.json, so viewers know the data on screen is illustrative.
export default function SampleDataWarning() {
  return (
    <div className="sample-warning" role="status">
      Showing sample data (data/sample_events.json) because data/events.json was not found.
    </div>
  )
}
