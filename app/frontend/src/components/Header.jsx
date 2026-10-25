// Header — app title and a one-line explanation of what the viewer is watching.
export default function Header() {
  return (
    <header className="header">
      <h1 className="header__title">Comm Ave Bike Lane Monitor</h1>
      <p className="header__subtitle">
        Replay demo — lane status and rider alerts played back in sync with the video.
      </p>
    </header>
  )
}
